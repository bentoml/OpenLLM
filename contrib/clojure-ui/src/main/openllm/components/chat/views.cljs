(ns openllm.components.chat.views
  (:require [re-frame.core :as rf]
            [openllm.components.chat.events :as events]
            [openllm.components.chat.subs :as subs]
            [openllm.components.side-bar.subs :as side-bar-subs]
            [openllm.components.side-bar.model-params.subs :as model-params-subs]
            [openllm.components.chat.views :as views]
            [openllm.api.components :as api-components]
            [reagent-mui.material.tooltip :refer [tooltip]]
            [reagent-mui.material.icon-button :refer [icon-button]]
            [reagent-mui.material.button :refer [button]]
            [reagent-mui.material.modal :refer [modal]]
            [reagent-mui.material.box :refer [box]]
            [reagent-mui.material.paper :refer [paper]]
            [reagent-mui.material.typography :refer [typography]]
            [reagent-mui.icons.design-services :as ds-icon]
            [reagent-mui.icons.send :as send-icon]
            [reagent.core :as r]))

(defn chat-input-field
  "The chat input field. The `on-submit` callback is called when the user
   presses the enter key without pressing the shift key.
   The same event is also dispatched in the `chat-controls` function, if
   the user clicks the send button."
  [on-submit]
  (let [chat-input-sub (rf/subscribe [::subs/chat-input-value])
        on-change #(rf/dispatch [::events/set-chat-input-value (.. % -target -value)])]
    (fn []
      [:textarea {:class "py-1 h-20 w-[calc(100%_-_80px)] block"
                  :style {:resize "none"}
                  :placeholder "Type your message..."
                  :value @chat-input-sub
                  :on-change on-change
                  :on-key-press (fn [e]
                                  (when (and (= (.-charCode e) 13) (not (.-shiftKey e)))
                                    (on-submit)))
                  :id "chat-input"}])))

(defn chat-controls
  "Aggregates the chat input field and the send button as well as the
   prompt layout button."
  []
  (let [llm-config (rf/subscribe [::model-params-subs/model-config])
        submit-prompt (rf/subscribe [::subs/prompt])
        on-submit-event [::events/on-send-button-click @submit-prompt @llm-config]]
    (fn chat-controls []
      [:form {:class "flex justify-end mr-2.5"}
       [chat-input-field #(rf/dispatch on-submit-event)]
       [:div {:class "grid grid-rows-2 ml-1.5 mr-0.5"}
        [:div {:class "items-start"}
         [tooltip {:title "Edit prompt layout"}
          [icon-button {:on-click #(rf/dispatch [::events/toggle-modal])
                        :color "primary"}
           [ds-icon/design-services]]]]
        [button {:on-click #(rf/dispatch on-submit-event)
                 :variant "outlined"
                 :end-icon (r/as-element [send-icon/send])
                 :style {:width "96px"}
                 :color "primary"}
         "Send"]]])))

(defn user->bubble-style
  "Produces additional style attributes for a chatbubble contingent upon
   the provided user.
   This can be done a lot smarter, but it works for now."
  [user]
  (str "p-2 rounded-xl border " (if (= user :model)
                                  "bg-gray-50 mr-10 rounded-bl-none border-gray-200"
                                  "bg-gray-300 ml-10 rounded-br-none border-gray-400")))

(defn user->text-style
  "Produces additional style attributes forthe text of a text message 
   upon the provided user.
   This can be done a lot smarter, but it works for now."
  [user]
  (str "whitespace-pre-wrap " (if (= user :model) "text-gray-700" "text-gray-950")))

(defn chat-message-entry
  "Displays a single chat message of the chat history.
   Will be used as a mapping function in the `chat-history` function. The collection
   being mapped is the entire chat history."
  [{:keys [user text]}]
  (let [display-user (if (= user :model) "System" "You")
        alignment (if (= user :model) "flex-row" "flex-row-reverse")]
    [:div {:class (str "flex " alignment " items-end my-2 w-full")}
     [:h3 {:class "font-bold text-lg mx-2"} display-user]
     [:div {:class (user->bubble-style user)}
      [:p {:class (user->text-style user)}
       text]]]))

(defn chat-history
  "The chat history. Transforms the chat history into DOM/hiccup elements by
   mapping the `chat-message-entry` function over the chat history."
  []
  (let [history (rf/subscribe [::subs/chat-history])]
    (fn chat-history []
      (into [:div {:class "px-8 flex flex-col items-center"}]
            (map chat-message-entry @history)))))

(defn prompt-layout-modal
  "The modal for editing the prompt layout. The modal is opened by the
   `toggle-modal` event and closed by the `toggle-modal` event. The modal
   is closed by clicking the save button or somewhere outside of the modal."
  []
  (let [modal-open? (rf/subscribe [::subs/modal-open?])
        prompt-layout-value (rf/subscribe [::subs/prompt-layout])
        on-change #(rf/dispatch [::events/set-prompt-layout (.. % -target -value)])]
    (fn []
      [modal {:open @modal-open?
              :on-close #(rf/dispatch [::events/toggle-modal])}
       [box {:style {:position "absolute"
                     :width 800,
                     :top "50%"
                     :left "50%"
                     :transform "translate(-50%, -50%)"}}
        [paper {:elevation 24
                :style {:padding "20px 30px"}}
         [typography {:variant "h5"} "Prompt Layout"]
         [:textarea {:class "pt-3 mt-1 w-full h-64 block border bg-gray-200"
                     :value @prompt-layout-value
                     :on-change on-change}]
         [:div {:class "mt-4 flex justify-end space-x-2"}
          [api-components/file-upload-button {:callback-event ::events/set-prompt-layout}]
          [button {:type "button"
                   :variant "outlined"
                   :on-click #(rf/dispatch [::events/toggle-modal])} "Save"]]]]])))

(defn chat-tab-contents
  "The component rendered if the chat tab is active. It contains the chat
   history, the chat input field and the chat controls."
  []
  (let [side-bar-open? (rf/subscribe [::side-bar-subs/side-bar-open?])]
    (fn []
      [:<>
       [prompt-layout-modal]
       [:div {:id "chat-history-container"
              :class "overflow-y-scroll mt-6 h-[calc(100%_-_116px)] w-full no-scrollbar"
              :style {:scrollBehavior "smooth"}}
        [:div
         [chat-history]]]
       [chat-controls]])))
