(ns openllm.components.model-selection.views
  (:require [openllm.components.model-selection.data :as data]
            [openllm.components.model-selection.subs :as subs]
            [openllm.components.model-selection.events :as events]
            [re-frame.core :as rf]))

(defn model-selection
  "The dropdowns selecting the model."
  []
  (let [model-type (rf/subscribe [::subs/selected-model-type])
        model-id (rf/subscribe [::subs/selected-model-id])]
    (fn []
      [:div {:class "px-5 mb-3 mt-1"}
       [:label {:class "text-black"} "Model-Type"
        (into [:select {:class "w-full pl-3 pr-10 py-1 mb-1"
                        :value @model-type
                        :on-change #(rf/dispatch [::events/set-model-type (-> % .-target .-value)])}]
              (map (fn [type]
                     [:option {:value type} type])
                   (keys data/models)))]
       [:label {:class "text-black"} "Model-ID"
        (into [:select {:class "w-full pl-3 pr-10 py-1"
                        :value @model-id
                        :on-change #(rf/dispatch [::events/set-model-id (-> % .-target .-value)])}]
              (map (fn [id]
                     [:option {:value id} (str id)])
                   (get-in data/models [@model-type :ids])))]])))
