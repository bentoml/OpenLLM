(ns openllm.components.playground.views
  (:require [openllm.components.playground.events :as events]
            [openllm.components.playground.subs :as subs]
            [openllm.api.components :as api-components]
            [openllm.components.common.views :as ui]
            [openllm.components.side-bar.model-params.subs :as model-params-subs]
            [re-frame.core :as rf]
            [reagent-mui.material.button :refer [button]]
            [reagent-mui.material.modal :refer [modal]]
            [reagent-mui.material.box :refer [box]]
            [reagent-mui.material.paper :refer [paper]]
            [reagent-mui.material.typography :refer [typography]]
            [reagent-mui.icons.send :as send-icon]
            [reagent.core :as r]))

(defn input-field
  "The input field for the prompt to send to the backend."
  []
  (let [value (rf/subscribe [::subs/playground-input-value])]
    (fn []
      [:textarea {:class "pt-3 w-full h-[calc(100%_-_74px)] block border"
                  :value @value
                  :on-change #(rf/dispatch [::events/set-prompt-input (.. % -target -value)])}])))

(defn input-field-controls
  "Control buttons for the input field, where the user enters his/her
   prompt."
  []
  (let [input-value (rf/subscribe [::subs/playground-input-value])
        llm-config (rf/subscribe [::model-params-subs/model-config])]
    (fn []
      [:div {:class "grid grid-cols-2"}
       [:div
        [api-components/file-upload
         {:callback-event ::events/set-prompt-input
          :class (str "w-7/12 mt-3 rounded cursor-pointer bg-gray-600 text-white hover:bg-gray-700 file:bg-gray-900 "
                      "file:cursor-pointer file:border-0 file:hover:bg-gray-950 file:mr-4 file:py-2 file:px-4 file:text-white")}]]
       [:div {:class "mt-3 flex justify-end space-x-2"}
        [button {:type "button"
                 :variant "outlined"
                 :on-click #(rf/dispatch [::events/set-prompt-input ""])} "Clear"]
        [button {:type "button"
                 :variant "outlined"
                 :end-icon (r/as-element [send-icon/send])
                 :on-click #(rf/dispatch [::events/on-send-button-click @input-value @llm-config])} "Send"]]])))

(defn response-area
  "The latest response retrieved from the backend will be displayed in this
   component.
   By default this is not visible, but it will be shown once the user has
   resized the input field. After that, resizing the browser window will no
   longer scale the input field."
  []
  (let [last-response (rf/subscribe [::subs/last-response])]
    (fn []
      [:div
       [ui/headline "Response" 0]
       [:textarea {:class "pt-3 mt-1 w-full h-64 block border bg-gray-200"
                   :value @last-response
                   :disabled true}]])))

(defn- response-modal
  "The modal that is shown to display server response in the playground view.
   It is opened and closed by the `::toggle-modal` event. The modal is closed
   by clicking the close button or somewhere outside of the modal."
  []
  (let [modal-open? (rf/subscribe [::subs/response-modal-open?])
        last-response (rf/subscribe [::subs/last-response])]
    (fn []
      [modal
       {:open @modal-open?
        :on-close #(rf/dispatch [::events/toggle-modal])
        :aria-labelledby "response-modal-title"
        :aria-describedby "response-modal-description"
        :actions [{:label "Close"
                   :on-click #(rf/dispatch [::events/toggle-modal])}]}
       [box {:style {:position "absolute"
                     :width 800,
                     :top "50%"
                     :left "50%"
                     :transform "translate(-50%, -50%)"}}
        [paper {:elevation 24
                :style {:padding "20px 30px"}}
         [typography {:id "response-modal-title"
                      :variant "h6"}
          "Response"]
         [typography {:id "response-modal-description"
                      :variant "body1"}
          @last-response]]]])))

(defn playground-tab-contents
  "This function aggregates all contents of the playground tab, and is
   called by the `tab-content` function residing in the `views` namespace
   directly."
  []
  [:div {:class "mt-6 pb-3.5 pr-3 h-full"}
   [response-modal]
   [input-field]
   [input-field-controls]
   [:div {:class "mt-6"}
    [response-area]]])
