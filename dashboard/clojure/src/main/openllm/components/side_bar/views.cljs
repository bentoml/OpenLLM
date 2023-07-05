(ns openllm.components.side-bar.views
  (:require [re-frame.core :as rf]
            [openllm.db :as db]
            [openllm.subs :as subs]
            [openllm.components.side-bar.events :as events] 
            [clojure.string :as str]))

(defn openllm-tag
  "The 'OpenLLM' tag in the very top right corner of the screen."
  []
  [:div {:class "flex items-center flex-shrink-0 px-6"}
   [:img {:class "h-11 w-auto" :src "./static/logo-light.svg" :alt "LOGO"}]
   [:span {:class "text-3xl font-bold text-gray-900 ml-2"} "OpenLLM"]])

(defn sidebar-group-headline
  "The headlines for the different groups in the sidebar are rendered using this component."
  [headline]
  [:h3 {:class "px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider" :id (str "sidebar-headline-" headline)} headline])

(defn parameter-slider-with-input
  "Renders a slider with an input field next to it."
  [name value]
  (let [min-max (name db/parameter-min-max)
        num-type? (or (str/includes? (str name) "num") (= name ::db/top_k))
        on-change #(rf/dispatch [::events/set-model-config-parameter name (if num-type?
                                                                            (parse-long (.. % -target -value))
                                                                            (parse-double (.. % -target -value)))])]
    [:div {:class "flex flex-row items-center"}
     [:span {:class "mr-2 text-xs text-gray-500"} (str (first min-max))]
     [:input {:type "range"
              :min (first min-max)
              :max (second min-max)
              :step (if num-type? 1 0.01)
              :value value
              :class "w-28 h-2 bg-gray-300 accent-pink-600"
              :on-change on-change}]
     [:span {:class "ml-2 text-xs text-gray-500"} (str (second min-max))]
     [:input {:type "number"
              :class "w-16 px-1 py-1 text-xs text-gray-700 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm ml-auto"
              :step (if num-type? 1 0.01)
              :value value
              :on-change on-change}]]))

(defn parameter-checkbox
  "Renders a checkbox."
  [name value]
  [:input {:type "checkbox"
           :checked value
           :class "h-4 w-4 text-pink-600 focus:ring-pink-500 border-gray-300 rounded"
           :on-change #(rf/dispatch [::events/set-model-config-parameter name (not value)])}])

(defn parameter-list-entry-value
  [name value]
  (cond
    (contains? db/parameter-min-max name) [parameter-slider-with-input name value]
    (boolean? value) [parameter-checkbox name value]
    :else
    [:input {:type "number"
             :class "px-1 py-1 text-xs text-gray-700 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm w-full"
             :value value
             :on-change #(rf/dispatch [::events/set-model-config-parameter name (int (.. % -target -value))])}]))

(defn parameter-list-entry
  "Renders a single parameter in the sidebar's parameter list."
  [[parameter-name parameter-value]]
  [:div {:class "flex flex-col px-3 py-2 text-sm font-medium text-gray-700"
         :key (str parameter-name)}
   [:span {:class "text-gray-500"} parameter-name]
   [:div {:class "mt-1"}
    [parameter-list-entry-value parameter-name parameter-value]]
   [:hr {:class "mt-6 border-1 border-black border-opacity-10"}]])

(defn parameter-list
  "Renders the parameters in the sidebar."
  []
  (let [model-config (rf/subscribe [::subs/model-config])]
    (fn parameter-list
      []
      (into [:div
             (map parameter-list-entry @model-config)]))))

(defn status-display
  "Displays the current service status at the bottom of the sidebar."
  [status-good?]
  (let [status-color (if status-good? "bg-green-500" "bg-red-500")
        status-text (if status-good? "Operational" "Degraded Performance")]
    [:div {:class "px-1 mt-1"}
     [:div {:class "mt-4"}
      [sidebar-group-headline "Service Status"]
      [:div {:class "space-y-1" :role "group" :aria-labelledby "service-status-headline"}
       [:a {:href "#" :class "group ml-3 flex items-center px-3 py-1 text-sm font-medium text-gray-700 rounded-md"}
        [:span {:class (str "w-2.5 h-2.5 mr-4 rounded-full " status-color) :aria-hidden "true"}]
        [:span {:class "truncate"} status-text]]]]]))

(defn side-bar
  "The render function of the toolbar on the very left of the screen"
  []
  [:div {:class "flex flex-col w-80 border-r border-gray-200 pt-5 pb-4 bg-gray-200"} ;; sidebar div + background
   [openllm-tag]
   [:hr {:class "my-5 border-1 border-black"}]
   [sidebar-group-headline "Parameters"]
   [:div {:class "my-4 h-0 flex-1 flex flex-col overflow-y-auto scrollbar"}
    [:div {:class "px-3 mt-3 relative inline-block text-left"}
     [parameter-list]]]
   [status-display true]])
