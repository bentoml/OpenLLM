(ns openllm.components.chat.views
  (:require [re-frame.core :as rf]
            [openllm.components.chat.events :as events]
            [openllm.components.chat.subs :as subs]
            [openllm.components.side-bar.subs :as side-bar-subs]
            [openllm.subs :as root-subs]
            [openllm.components.chat.views :as views]
            [openllm.api.persistence :as persistence]
            [openllm.components.common.views :as ui]))

(defn chat-controls
  "The chat input field and the send button."
  []
  (let [chat-input-sub (rf/subscribe [::subs/chat-input-value])
        llm-config (rf/subscribe [::root-subs/model-config])
        on-change #(rf/dispatch [::events/set-chat-input-value (.. % -target -value)])
        on-send-click #(rf/dispatch [::events/on-send-button-click @chat-input-sub @llm-config])]
    (fn chat-controls []
      [:form {:class "flex items-center justify-between"}
       [:textarea {:class "py-1 w-[calc(100%_-_80px)] appearance-none block border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-gray-500 focus:border-gray-500 sm:text-sm h-20"
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
       [:div {:class "grid grid-rows-1"}
        [ui/tooltip
         [:button {:class "bg-gray-400 hover:bg-gray-600 text-white py-1 px-2 rounded text-xl"
                   :on-click #(js/window.alert "not implemented")
                   :type "button"} "📋"]
         "Edit prompt layout"]
        [:button {:class "mt-1 px-4 py-2 text-white bg-gray-600 rounded-md hover:bg-gray-700 focus:outline-none block"
                  :on-click on-send-click
                  :type "button"} "Send"]]])))

(defn user->extra-bubble-style
  "Produces additional style attributes for a chatbubble contingent upon
   the provided user."
  [user]
  (if (= user :model)
     "bg-gray-50 mr-10 rounded-bl-none border-gray-200"
     "bg-gray-300 ml-10 rounded-br-none border-gray-400"))

(defn chat-history
  "The chat history."
  []
  (let [history (rf/subscribe [::subs/chat-history])]
    (fn chat-history []
      (into [:div {:class "px-8 flex flex-col items-center"}]
            (map (fn [{:keys [user text]}]
                   (let [display-user (if (= user :model) "System" "You")
                         alignment (if (= user :model) "flex-row" "flex-row-reverse")]
                     [:div {:class (str "flex " alignment " items-end my-2 w-full")}
                      [:h3 {:class "font-bold text-lg mx-2"} display-user]
                      [:div {:class (str "p-2 rounded-xl border " (user->extra-bubble-style user))}
                       [:p {:class (if (= user :model) "text-gray-700" "text-gray-950")} text]]]))
                 @history)))))

(defn clear-history-button
  "The button to clear the chat history."
  []
  (let [side-bar-open? (rf/subscribe [::side-bar-subs/side-bar-open?])]
    (fn []
      [:div {:class "fixed top-24 h-[calc(100%_-_220px)] pr-1"
             :style {:zIndex "99"
                     :right (if @side-bar-open?
                              "20rem"
                              "0")}}
       [ui/tooltip
        [:button {:class "bg-gray-200 hover:bg-black rounded block text-xl font-bold border border-gray-500"
                  :on-click #(do (rf/dispatch [::events/clear-chat-history])
                                 (rf/dispatch [::persistence/clear-chat-history]))} "✖"]
        "Clear chat history"]])))

(defn chat-tab-contents
  "The component rendered if the chat tab is active."
  []
  (let [chat-empty? (rf/subscribe [::subs/chat-history-empty?])
        side-bar-open? (rf/subscribe [::side-bar-subs/side-bar-open?])]
    (fn []
      [:<>
       [:div {:id "chat-history-container"
              :class "overflow-y-scroll mt-6 h-[calc(100%_-_220px)] w-full no-scrollbar"
              :style {:scrollBehavior "smooth"}}
        [:div
         [chat-history]]]
       (when (not @chat-empty?)
         [clear-history-button])
       [:div {:class (str "bottom-1 fixed w-[calc(100%_-_200px)] mb-2"
                          (if @side-bar-open?
                            " w-[calc(100%_-_350px)]"
                            " w-[calc(100%_-_30px)]"))}
        [chat-controls]]])))
