(ns openllm.components.model-selection.views)

(defn model-selection
  "The dropdowns selecting the model."
  []
  [:div {:class "mt-1 px-2 py-2 items-center relative rounded-md shadow-md shadow-gray-200 border-gray-200 border-solid border grid grid-cols-4"}
   [:label {:class "text-end pr-6"} "Model-Type"]
   [:select {:class "w-2/3 pl-3 pr-10 py-1"}
    [:option {:value "flan-type"} "FLAN-T5"]]
   [:label {:class "text-end pr-6"} "Model-ID"]
   [:select {:class "w-2/3 pl-3 pr-10 py-1"}
    [:option {:value "flan-id"} "google/flan-t5-large"]]])
