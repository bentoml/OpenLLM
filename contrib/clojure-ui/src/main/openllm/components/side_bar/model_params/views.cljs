(ns openllm.components.side-bar.model-params.views
  (:require [openllm.components.side-bar.model-params.db :as db]
            [openllm.components.side-bar.model-params.subs :as subs]
            [openllm.components.side-bar.model-params.events :as events]
            [reagent-mui.material.input :refer [input]]
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
     [input {:type "number"
             :class "w-10 text-center border"
             :input-props {:style {:text-align "center"
                                   :background-color "#ffffff"
                                   :height "12px"}}
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
  [input {:type "number"
          :class "w-16 border"
          :value value
          :input-props {:style {:padding "2px"
                                :background-color "#ffffff"
                                :height "24px"}}
          :on-change #(rf/dispatch [::events/set-model-config-parameter id (parse-long (.. % -target -value))])}])

(defn parameter-list-entry
  "Renders a single parameter in the sidebar's parameter list. Used as a mapping function
   on the collection of all parameters."
  [[id {:keys [value name]}]]
  (let [display-type (get-in db/parameter-meta-data [id :display-type])]
    [:<>
     [:div {:class "flex flex-col px-2 pt-1"}
      [:label {:class "flex w-full text-xs justify-between"}
       [:div {:class "self-center"} name]
       (condp = display-type
         :slider
         [parameter-small-input id value]
         :binary
         [parameter-checkbox id value]
         :field
         [parameter-number id value])]
      (when (= :slider display-type)
        [:div {:class "mb-0.5"}
         [parameter-slider id value]])]
     [:hr {:class "mt-1 border-1 border-gray-100 last:border-0 last:mt-0 last:-mb-1.5"}]]))

(defn parameter-list
  "Renders the parameters in the sidebar. The parameters are retrieved from the
   `human-readable-config` subscription."
  []
  (let [model-config (rf/subscribe [::subs/human-readable-config])]
    (fn []
      (let [basic-params (filterv (fn [[id _]] (not (get-in db/parameter-meta-data [id :advanced-opt]))) @model-config) ;; TODO: views shouldn't make such heave calculations
            advanced-params (filterv (fn [[id _]] (get-in db/parameter-meta-data [id :advanced-opt])) @model-config)]
        [:<>
         (into [:<>] (map parameter-list-entry basic-params))
         [:div {:class "mt-2 -mx-1.75"}
          [accordion {:square true
                      :class "w-full"
                      :elevation 0
                      :style {:background-color "#fafafa"}}
           [accordion-summary {:expand-icon (r/as-element [expand-more])}
            [typography "Advanced"]]
           [accordion-details {:class "-mt-1.5 -mx-3"}
            (into [:<>] (map parameter-list-entry advanced-params))]]]]))))
