(ns openllm.components.chat.views
  (:require [re-frame.core :as rf]
            [openllm.components.chat.events :as events]
            [openllm.components.chat.subs :as subs]
            [openllm.subs :as root-subs]
            [openllm.components.chat.views :as views]))

(defn chat-controls
  "The chat input field and the send button."
  []
  (let [chat-input-sub (rf/subscribe [::subs/chat-input-value])
        llm-config (rf/subscribe [::root-subs/model-config])
        on-change #(rf/dispatch [::events/set-chat-input-value (.. % -target -value)])
        on-send-click #(rf/dispatch [::events/on-send-button-click @chat-input-sub @llm-config])]
    (fn chat-controls []
       [:form {:class "flex items-center justify-between"
               :on-submit #(do % (on-send-click)
                               (.preventDefault %))}
        [:textarea {:class "py-1 w-[calc(100%_-_80px)] appearance-none block border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm"
                    :style {:resize "none"}
                    :type "text" :placeholder "Type your prompt..."
                    :value @chat-input-sub
                    :on-change on-change
                    :id "chat-input"
                    :auto-complete "off"
                    :auto-correct "off"}]
        [:button {:class "ml-2 px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none"
                  :on-click on-send-click
                  :type "button"} "Send"]])))

(defn chat-history
  "The chat history."
  []
  (let [history (rf/subscribe [::subs/chat-history])]
    (fn chat-history []
      (into [:div {:class "px-4"}]
            (map (fn [{:keys [user text]}]
                   (let [diplay-user (if (= user :model) "Model" "You")
                         color (if (= user :model) "bg-gray-200" "bg-blue-200")]
                     [:div {:class (str "p-2 rounded-lg mb-2 " color)}
                      [:h3 {:class "font-bold text-lg"} diplay-user]
                      [:p {:class "text-gray-700"} text]]))
                 @history)))))

(defn chat-tab
  "The component rendered if the chat tab is active."
  []
  [:div {:id "chat-history-container"
         :class "overflow-y-scroll mt-6 h-[calc(100%_-_200px)] w-full no-scrollbar"
         :style {:scrollBehavior "smooth"}}
   [:div
    [chat-history]]
   [:div {:class "bottom-1 fixed w-[calc(100%_-_380px)]"}  
    [chat-controls]]])
