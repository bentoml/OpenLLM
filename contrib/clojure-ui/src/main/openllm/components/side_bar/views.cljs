(ns openllm.components.side-bar.views
  (:require [re-frame.core :as rf]
            [openllm.components.side-bar.db :as db]
            [openllm.components.model-selection.views :as model-selection-view]
            [openllm.components.side-bar.subs :as subs]
            [openllm.components.side-bar.events :as events]
            [openllm.components.common.views :as ui]
            [clojure.string :as str]
            [reagent-mui.material.collapse :refer [collapse]]))

(defn num-type?
  "Returns true if the parameter is a number, false otherwise."
  [id]
  (or (str/includes? (str id) "num") (= id ::db/top_k)))

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
  [:div {:class "flex flex-col px-2 py-1"}
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
      (into [:<>] (map parameter-list-entry @model-config)))))

(defn side-bar-with-mui-collapse
  "The sidebar wrapped with a Material UI Collapse component."
  []
  (let [side-bar-open? (rf/subscribe [::subs/side-bar-open?])]
    (fn []
      [collapse {:in @side-bar-open?
                 :orientation "horizontal"
                 :class "flex flex-col h-full w-80"}
       [:div {:class "flex flex-col w-80 h-screen border-l border-gray-200 pt-0.5 pb-4 bg-gray-50"}
        [model-selection-view/model-selection]
        [:hr {:class "mb-2 border-1 border-gray-200"}]
        [ui/headline "Parameters"]
        [:div {:class "my-0 h-0 flex-1 flex flex-col overflow-y-auto scrollbar"}
         [:div {:class "px-3 mt-0 relative inline-block text-left"}
          [parameter-list]]]]])))

(defn side-bar
  "The render function of the toolbar on the very left of the screen"
  []
  [:div {:class "hidden lg:flex lg:flex-shrink-0"}
   [side-bar-with-mui-collapse]])
