(ns openllm.components.chat.views
  (:require [re-frame.core :as rf]
            [openllm.components.chat.events :as events]
            [openllm.components.chat.subs :as subs]
            [openllm.subs :as root-subs]
            [openllm.components.chat.views :as views]
            [openllm.api.persistence :as persistence]
            [openllm.api.components :as ui]))

(defn chat-controls
  "The chat input field and the send button."
  []
  (let [chat-input-sub (rf/subscribe [::subs/chat-input-value])
        llm-config (rf/subscribe [::root-subs/model-config])
        on-change #(rf/dispatch [::events/set-chat-input-value (.. % -target -value)])
        on-send-click #(rf/dispatch [::events/on-send-button-click @chat-input-sub @llm-config])]
    (fn chat-controls [] 
      [:form {:class "flex items-center justify-between"}
       [:textarea {:class "py-1 w-[calc(100%_-_80px)] appearance-none block border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm h-20"
                   :style {:resize "none"}
                   :type "text" :placeholder "Type your message..."
                   :value @chat-input-sub
                   :on-change on-change
                   :on-key-press (fn [e]
                                   (when (and (= (.-charCode e) 13) (not (.-shiftKey e)))
                                     (on-send-click)))
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
      (into [:div {:class "px-4 flex flex-col items-center"}]
            (map (fn [{:keys [user text]}]
                   (let [display-user (if (= user :model) "System" "You")
                         color (if (= user :model) "bg-gray-200" "bg-pink-300")
                         alignment (if (= user :model) "flex-row" "flex-row-reverse")
                         margin (if (= user :model) "mr-10" "ml-10")]
                     [:div {:class (str "flex " alignment " items-center my-2 w-4/5")}
                      [:h3 {:class "font-bold text-lg mx-2"} display-user]
                      [:div {:class (str "p-2 rounded-xl " color " " margin)}
                       [:p {:class "text-gray-700"} text]]]))
                 @history)))))

(defn clear-history-button
  "The button to clear the chat history."
  []
  [:div {:class "fixed top-32 h-[calc(100%_-_220px)] pr-2 pt-2"
         :style {:zIndex "9999"
                 :right "21.1rem"}}
   [ui/tooltip
    [:button {:class "bg-pink-600 hover:bg-pink-800 text-white py-2 px-4 rounded block text-xl"
              :on-click #(do (rf/dispatch [::events/clear-chat-history])
                             (rf/dispatch [::persistence/clear-chat-history]))} "🗑️"]
    "Click to clear chat history"]])

(defn chat-tab-contents
  "The component rendered if the chat tab is active."
  []
  [:<>
   [:div {:id "chat-history-container"
          :class "overflow-y-scroll mt-6 h-[calc(100%_-_220px)] w-full no-scrollbar"
          :style {:scrollBehavior "smooth"}}
    [:div
     [chat-history]]]
   [clear-history-button]
   [:div {:class "bottom-1 fixed w-[calc(100%_-_380px)] mb-2"}
    [chat-controls]]])
