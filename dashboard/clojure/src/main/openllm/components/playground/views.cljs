(ns openllm.components.playground.views
  (:require [openllm.components.playground.events :as events]
            [openllm.components.playground.subs :as subs]
            [openllm.components.model-selection.views :as model-selection-view]
            [openllm.subs :as root-subs]
            [re-frame.core :as rf]
            [openllm.api.components :as api-components]))

(defn input-field
  "The input field for the prompt to send to the backend."
  []
  (let [value (rf/subscribe [::subs/playground-input-value])]
    (fn []
      [:textarea {:class "pt-3 appearance-none w-full h-64 block border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm"
                  :value @value
                  :on-change #(rf/dispatch [::events/set-prompt-input (.. % -target -value)])}])))    


(defn input-field-controls
  "Control buttons for the input field, where the user enters his/her
   prompt."
  []
  (let [input-value (rf/subscribe [::subs/playground-input-value])
        llm-config (rf/subscribe [::root-subs/model-config])]
    (fn []
      [:div {:class "grid grid-cols-2"}
       [:div
        [api-components/file-upload
         {:callback-event ::events/set-prompt-input
          :class (str "w-7/12 mt-3 shadow-sm rounded-md cursor-pointer bg-blue-600 text-white hover:bg-blue-700 focus:z-10 file:bg-blue-900 "
                      "file:cursor-pointer file:border-0 file:hover:bg-blue-950 file:mr-4 file:py-2 file:px-4 file:text-white")}]]
       [:div {:class "mt-3 flex justify-end"}
        [:button {:class "px-4 py-2 mr-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none"
                  :type "button"
                  :on-click #(rf/dispatch [::events/set-prompt-input ""])} "Clear"]
        [:button {:class "px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none"
                  :type "button"
                  :on-click #(rf/dispatch [::events/on-send-button-click @input-value @llm-config])} "Send"]]])))

(defn response-area
  "The latest response retrieved from the backend will be displayed in
   this component."
  []
  (let [last-response (rf/subscribe [::subs/last-response])]
    (fn []
      [:div
       [:label {:class "text-end pr-6"} "Response"]
       [:textarea {:class "pt-3 appearance-none w-full h-64 block border bg-gray-200 border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm"
                   :value @last-response
                   :disabled true}]])))

(defn playground-tab-contents
  "This function aggregates all contents of the playground tab, and is
   called by the `tab-content` function residing in the `views` namespace
   directly."
  []
  [:div {:class "mt-6 px-4"}
   [model-selection-view/model-selection]
   [:div {:class "mt-4"}
    [input-field]
    [input-field-controls]]
   [:div {:class "mt-6"}
    [response-area]]])
