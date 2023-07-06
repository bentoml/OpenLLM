(ns openllm.components.chat.views
  (:require [re-frame.core :as rf]
            [openllm.components.chat.events :as events]
            [openllm.components.chat.subs :as subs]
            [openllm.components.side-bar.subs :as side-bar-subs]
            [openllm.subs :as root-subs]
            [openllm.components.chat.views :as views]
            [openllm.api.persistence :as persistence]
            [reagent-mui.material.icon-button :refer [icon-button]]
            [reagent-mui.material.button :refer [button]]
            [reagent-mui.icons.delete :as delete-icon]
            [reagent-mui.icons.design-services :as ds-icon]
            [reagent-mui.icons.send :as send-icon]
            [reagent.core :as r]))

(defn chat-controls
  "The chat input field and the send button."
  []
  (let [chat-input-sub (rf/subscribe [::subs/chat-input-value])
        llm-config (rf/subscribe [::root-subs/model-config])
        on-change #(rf/dispatch [::events/set-chat-input-value (.. % -target -value)])
        on-send-click #(rf/dispatch [::events/on-send-button-click @chat-input-sub @llm-config])]
    (fn chat-controls []
      [:form {:class "flex justify-end"}
       [:textarea {:class "py-1 h-20 w-[calc(100%_-_80px)] block"
                   :style {:resize "none"}
                   :placeholder "Type your message..."
                   :value @chat-input-sub
                   :on-change on-change
                   :on-key-press (fn [e]
                                   (when (and (= (.-charCode e) 13) (not (.-shiftKey e)))
                                     (on-send-click)))
                   :id "chat-input"}]
       [:div {:class "grid grid-rows-2 ml-1.5"}
        [:div {:class "items-start"}
         [icon-button {:on-click #(js/window.alert "not implemented")
                       :color "primary"}
          [ds-icon/design-services]]] 
        [button {:on-click on-send-click
                 :variant "outlined"
                 :end-icon (r/as-element [send-icon/send])
                 :color "primary"} "Send"]]])))

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
        [icon-button {:on-click #(do (rf/dispatch [::events/clear-chat-history])
                                     (rf/dispatch [::persistence/clear-chat-history]))
                      :size "small"
                      :color "error"}
         [delete-icon/delete]]])))

(defn chat-tab-contents
  "The component rendered if the chat tab is active."
  []
  (let [chat-empty? (rf/subscribe [::subs/chat-history-empty?])
        side-bar-open? (rf/subscribe [::side-bar-subs/side-bar-open?])]
    (fn []
      [:div
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
