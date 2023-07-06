(ns openllm.components.model-selection.views)

(defn model-selection
  "The dropdowns selecting the model."
  []
  [:div {:class "px-5 mb-3 mt-1"}
   [:label {:class "text-black"} "Model-Type"
    [:select {:class "w-full pl-3 pr-10 py-1 mb-1"}
     [:option {:value "flan-type"} "FLAN-T5"]]]
   [:label {:class "text-black"} "Model-ID"
    [:select {:class "w-full pl-3 pr-10 py-1"}
     [:option {:value "flan-id"} "google/flan-t5-large"]]]])
