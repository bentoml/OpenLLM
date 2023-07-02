(ns openllm.components.playground.views
  (:require [openllm.components.playground.events :as events]
            [openllm.components.playground.subs :as subs]
            [openllm.subs :as root-subs]
            [re-frame.core :as rf]))

(defn model-selection
  "The dropdowns selecting the model."
  []
  [:div {:class "mt-1 px-2 py-2 items-center relative rounded-md shadow-sm grid grid-cols-4"}
   [:label {:class "text-end pr-6"} "Model-Type"]
   [:select {:class "w-2/3 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm rounded-md"}
    [:option {:value "flan-type"} "FLAN-T5"]]
   [:label {:class "text-end pr-6"} "Model-ID"]
   [:select {:class "w-2/3 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm rounded-md"}
    [:option {:value "flan-id"} "google/flan-t5-large"]]])

(defn input-field
  "The input field for the prompt."
  []
  (let [input-value (rf/subscribe [::subs/playground-input-value])
        llm-config (rf/subscribe [::root-subs/model-config])]
    (fn []
      [:div {:class "mt-2"}
       [:textarea {:class "pt-3 appearance-none w-full h-64 block border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm"
                   :value @input-value
                   :on-change #(rf/dispatch [::events/set-prompt-input (.. % -target -value)])}]
       [:div {:class "mt-3 flex justify-end"}
        [:button {:class "px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none"
                  :type "button"
                  :on-click #(rf/dispatch [::events/on-send-button-click @input-value @llm-config])} "Send"]
        [:button {:class "ml-2 px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none"
                  :type "button"
                  :on-click #(rf/dispatch [::events/set-prompt-input ""])} "Clear"]]])))

(defn response-area
  "The latest response will be displayed in the component."
  []
  (let [last-response (rf/subscribe [::subs/last-response])]
    (fn []
      [:div
       [:label {:class "text-end pr-6"} "Response"]
       [:textarea {:class "pt-3 appearance-none w-full h-64 block border bg-gray-200 border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm"
                   :value @last-response
                   :disabled true}]])))

(defn playground-tab
  "Contents of the playground tab."
  []
  [:div {:class "mt-6 px-4"}
   [model-selection]
   [input-field]
   [:div {:class "mt-6"}
    [response-area]]])
