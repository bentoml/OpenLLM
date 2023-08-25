(ns openllm.components.side-bar.model-selection.views
  (:require [openllm.components.side-bar.model-selection.subs :as subs]
            [openllm.components.side-bar.model-selection.events :as events]
            [reagent-mui.material.select :refer [select]]
            [re-frame.core :as rf]))

(defn model-selection
  "The dropdowns selecting the model. The `model-type` dropdown is populated
   with the available `model-types`, the `model-id` dropdown is populated with
   the available `model-ids` for the currently selected `model-type`."
  []
  (let [model-type (rf/subscribe [::subs/selected-model-type])
        model-id (rf/subscribe [::subs/selected-model-id])
        all-model-types (rf/subscribe [::subs/all-model-types])
        all-model-ids (rf/subscribe [::subs/all-model-ids])]
    (fn []
      [:div {:class "px-5 mb-3 mt-1"}
       [:label {:class "text-black"} "Model"
        (into [select {:class "w-full h-8 mb-1"
                       :value @model-type
                       :read-only true}]
              (map #(do [:option {:value %} %])
                   @all-model-types))]
       [:label {:class "text-black"} "Model ID"
        (into [select {:class "w-full h-8"
                       :value @model-id
                       :read-only true}]
              (map #(do [:option {:value %} (str %)])
                   @all-model-ids))]])))
