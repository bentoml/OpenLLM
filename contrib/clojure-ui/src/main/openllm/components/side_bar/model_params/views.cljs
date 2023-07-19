(ns openllm.components.side-bar.model-params.views
  (:require [openllm.components.side-bar.model-params.db :as db]
            [openllm.components.side-bar.model-params.subs :as subs] 
            [openllm.components.side-bar.model-params.events :as events]
            [clojure.string :as str]
            [re-frame.core :as rf]))

(defn num-type?
  "Returns true if the parameter is a number, false otherwise."
  [id]
  (or (str/includes? (str id) "num")
      (= id :top_k)))

(defn parameter-slider
  "Renders a slider with an input field next to it. The num-type logic needs to be
   revamped big time xx"
  [id value]
  (let [min-max (id db/parameter-constraints)
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
  "Renders a small input field, used in combination with the sliders. The num-type logic
   needs to be revamped big time xx"
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
            :on-change #(rf/dispatch [::events/set-model-config-parameter id (parse-long (.. % -target -value))])}]])

(defn parameter-list-entry
  "Renders a single parameter in the sidebar's parameter list. Used as a mapping function
   on the collection of all parameters."
  [[id {:keys [value name]}]]
  [:div {:class "flex flex-col px-2 py-1"}
   [:label {:class "flex w-fit text-xs"}
    name
    (when (contains? db/parameter-constraints id)
      [parameter-small-input id value])
    (when (boolean? value)
      [parameter-checkbox id value])
    (when (and (not (contains? db/parameter-constraints id)) (not (boolean? value)))
      [parameter-number id value])]
   (when (contains? db/parameter-constraints id)
     [:div {:class "mt-0.5"}
      [parameter-slider id value]])
   [:hr {:class "mt-1.5 border-1 border-gray-100"}]])

(defn parameter-list
  "Renders the parameters in the sidebar. The parameters are retrieved from the
   `human-readable-config` subscription."
  []
  (let [model-config (rf/subscribe [::subs/human-readable-config])]
    (fn []
      (into [:<>] (map parameter-list-entry @model-config)))))
