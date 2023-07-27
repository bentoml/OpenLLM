(ns openllm.components.side-bar.model-params.views
  (:require [openllm.components.side-bar.model-params.db :as db]
            [openllm.components.side-bar.model-params.subs :as subs] 
            [openllm.components.side-bar.model-params.events :as events]
            [reagent-mui.material.accordion :refer [accordion]]
            [reagent-mui.material.accordion-details :refer [accordion-details]]
            [reagent-mui.material.accordion-summary :refer [accordion-summary]]
            [reagent-mui.material.typography :refer [typography]]
            [reagent-mui.icons.expand-more :refer [expand-more]]
            [clojure.string :as str]
            [re-frame.core :as rf]
            [reagent.core :as r]))

(defn parameter-slider
  "Renders a slider with an input field next to it. The num-type logic needs to be
   revamped big time xx"
  [id value]
  (let [min-max (get-in db/parameter-meta-data [id :val-constraint])
        num-type? (= int? (get-in db/parameter-meta-data [id :type-pred]))
        on-change #(rf/dispatch [::events/set-model-config-parameter id (.. % -target -value)])]
    [:div {:class "flex flex-row items-center w-full"}
     [:input {:type "range"
              :min (first min-max)
              :max (second min-max)
              :step (if num-type? 1 0.01)
              :value value
              :class "w-full mt-2 mb-1"
              :on-change on-change}]]))

(defn parameter-small-input
  "Renders a small input field, used in combination with the sliders. The num-type logic
   needs to be revamped big time xx"
  [id value]
  (let [on-change #(rf/dispatch [::events/set-model-config-parameter id (.. % -target -value)])
        num-type? (= int? (get-in db/parameter-meta-data [id :type-pred]))]
    [:input {:type "number"
             :class "display-none absolute right-5 w-12 px-0 py-0 pr-0.5 text-xs text-center"
             :step (if num-type? 1 0.01)
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
            :on-change #(rf/dispatch [::events/set-model-config-parameter id (parse-long (.. % -target -value))])}]])

(defn parameter-list-entry
  "Renders a single parameter in the sidebar's parameter list. Used as a mapping function
   on the collection of all parameters."
  [[id {:keys [value name]}]]
  (let [display-type (get-in db/parameter-meta-data [id :display-type])]
    [:div {:class "flex flex-col px-2 py-1"}
     [:label {:class "flex w-fit text-xs"}
      name
      (when (= :slider display-type)
        [parameter-small-input id value])
      (when (= :binary display-type)
        [parameter-checkbox id value])
      (when (= :field display-type)
        [parameter-number id value])]
     (when (= :slider display-type)
       [:div {:class "mt-0.5"}
        [parameter-slider id value]])
     [:hr {:class "mt-1.5 border-1 border-gray-100"}]]))

(defn parameter-list
  "Renders the parameters in the sidebar. The parameters are retrieved from the
   `human-readable-config` subscription."
  []
  (let [model-config (rf/subscribe [::subs/human-readable-config])
        basic-params (filterv (fn [[id _]] (not (get-in db/parameter-meta-data [id :advanced-opt]))) @model-config)
        advanced-params (filterv (fn [[id _]] (get-in db/parameter-meta-data [id :advanced-opt])) @model-config)]
    (fn []
      [:<>
       (into [:<>] (map parameter-list-entry basic-params))
       [:div {:class "mt-2 -mx-1.75"}
        [accordion {:square true
                    :class "w-full"}
         [accordion-summary {:expand-icon (r/as-element [expand-more])}
          [typography "Advanced"]]
         [accordion-details {:class "mt-2 -mx-3"}
          (into [:<>] (map parameter-list-entry advanced-params))]]]])))
