(ns openllm.components.side-bar.views
  (:require [re-frame.core :as rf]
            [openllm.db :as db]
            [openllm.components.side-bar.subs :as subs]
            [openllm.components.side-bar.events :as events]
            [openllm.components.common.views :as ui]
            [clojure.string :as str]))

(defn num-type?
  "Returns true if the parameter is a number, false otherwise."
  [id]
  (or (str/includes? (str id) "num") (= id ::db/top_k)))

(defn openllm-tag
  "The 'OpenLLM' tag in the very top right corner of the screen."
  []
  [:div {:class "flex items-center flex-shrink-0 px-6"}
   [:img {:class "ml-10 h-11 w-auto" :src "./static/logo-light.svg" :alt "LOGO"}]
   [:span {:class "text-3xl font-bold text-gray-900 ml-2"} "OpenLLM"]])

(defn parameter-slider
  "Renders a slider with an input field next to it."
  [id value]
  (let [min-max (id db/parameter-min-max)
        on-change #(rf/dispatch [::events/set-model-config-parameter id (if (num-type? id)
                                                                            (parse-long (.. % -target -value))
                                                                            (parse-double (.. % -target -value)))])]
    [:div {:class "flex flex-row items-center w-full"}
     [:input {:type "range"
              :min (first min-max)
              :max (second min-max)
              :step (if (num-type? id) 1 0.01)
              :value value
              :class "w-full mt-2 mb-1"
              :on-change on-change}]]))

(defn parameter-small-input
  "Renders a small input field, used in combination with the sliders."
  [id value]
  (let [on-change #(rf/dispatch [::events/set-model-config-parameter id (if (num-type? id)
                                                                          (parse-long (.. % -target -value))
                                                                          (parse-double (.. % -target -value)))])]
    [:input {:type "number"
             :class "display-none absolute right-5 w-12 px-0 py-0 pr-0.5 text-xs text-center"
             :step (if (num-type? id) 1 0.01)
             :value value
             :on-change on-change}]))

(defn parameter-checkbox
  "Renders a checkbox."
  [id value]
  [:input {:type "checkbox"
           :class "ml-6 mt-1"
           :checked value
           :on-change #(rf/dispatch [::events/set-model-config-parameter id (not value)])}])

(defn parameter-number
  "Renders a number input field."
  [id value]
  [:div {:class "absolute right-5 -mt-0.5"}
   [:input {:type "text"
            :class "px-1 py-0 text-xs rounded w-16"
            :value value
            :on-change #(rf/dispatch [::events/set-model-config-parameter id (.. % -target -value)])}]])

(defn parameter-list-entry
  "Renders a single parameter in the sidebar's parameter list."
  [[id {:keys [value name]}]]
  [:div {:class "flex flex-col px-2 py-1"
         :key (str id)}
   [:label {:class "flex w-fit text-xs"}
    name
    (when (contains? db/parameter-min-max id)
      [parameter-small-input id value])
    (when (boolean? value)
      [parameter-checkbox id value])
    (when (and (not (contains? db/parameter-min-max id)) (not (boolean? value)))
      [parameter-number id value])]
   (when (contains? db/parameter-min-max id)
     [:div {:class "mt-0.5"} [parameter-slider id value]])
   [:hr {:class "mt-1.5 border-1 border-gray-100"}]])

(defn parameter-list
  "Renders the parameters in the sidebar."
  []
  (let [model-config (rf/subscribe [::subs/human-readable-config])]
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
      [ui/headline "Service Status"]
      [:div {:class "space-y-1" :role "group"}
       [:a {:href "#" :class "group ml-3 flex items-center px-3 py-1 text-sm font-medium text-gray-700 rounded-md"}
        [:span {:class (str "w-2.5 h-2.5 mr-4 rounded-full " status-color) :aria-hidden "true"}]
        [:span {:class "truncate"} status-text]]]]]))

(defn sidebar-expanded
  "The render function of the sidebar when it is expanded."
  []
  [:div {:class "flex flex-col w-80 border-r border-gray-200 pt-5 pb-4 bg-gray-50"} ;; sidebar div + background
   [openllm-tag]
   [:hr {:class "my-5 border-1 border-black"}]
   [ui/headline "Parameters"]
   [:div {:class "my-4 h-0 flex-1 flex flex-col overflow-y-auto scrollbar"}
    [:div {:class "px-3 mt-3 relative inline-block text-left"}
     [parameter-list]]]
   [status-display true]])

(defn sidebar-minimized
  "The render function of the sidebar when it is minimized."
  [open?]
  [:div {:class "mt-5 h-7 float-left bg-gray-950 hover:bg-gray-800 text-xl rounded rounded-l-2xl rounded-r-none"}
   [:button {:class "text-xl text-white font-bold"
             :on-click #(rf/dispatch [::events/toggle-side-bar])}
    (if open? "→" "←")]])

(defn side-bar
  "The render function of the toolbar on the very left of the screen"
  []
  (let [side-bar-open? (rf/subscribe [::subs/side-bar-open?])]
    (fn []
      [:div {:class "hidden lg:flex lg:flex-shrink-0"}
       [sidebar-minimized @side-bar-open?]
       (when @side-bar-open? [sidebar-expanded])])))
