(ns openllm.components.nav-bar.views
  (:require [re-frame.core :as rf]
            [openllm.components.nav-bar.subs :as subs]
            [openllm.components.side-bar.events :as side-bar-events]
            [openllm.components.side-bar.subs :as side-bar-subs]
            [openllm.components.chat.events :as chat-events]
            [openllm.api.persistence :as persistence]
            [reagent-mui.icons.chat :as chat-icon]
            [reagent-mui.icons.brush :as brush-icon]
            [reagent-mui.material.app-bar :refer [app-bar]]
            [reagent-mui.material.toolbar :refer [toolbar]]
            [reagent-mui.material.button :refer [button]]
            [reagent-mui.material.icon-button :refer [icon-button]]
            [reagent-mui.material.typography :refer [typography]]
            [reagent-mui.icons.keyboard-double-arrow-right :as right-icon]
            [reagent-mui.icons.keyboard-double-arrow-left :as left-icon]
            [reagent-mui.icons.delete-forever :as delete-icon]
            [reagent.core :as r]))

(defn- collapse-side-bar-button
  "The collapse side bar button. Only visible when the side bar is open."
  []
  (let [side-bar-open? (rf/subscribe [::side-bar-subs/side-bar-open?])]
    (fn []
      [:div {:class "w-full flex justify-end -mr-8"}
       [icon-button {:on-click #(rf/dispatch [::side-bar-events/toggle-side-bar])
                     :color "inherit"}
        (if @side-bar-open?
          [right-icon/keyboard-double-arrow-right]
          [left-icon/keyboard-double-arrow-left])]])))

(defn- clear-chat-history-button
  "The clear chat history button. Only visible when the chat history is not empty
   and we are in the chat history screen."
  []
  (let [active-screen (rf/subscribe [:screen-id])
        chat-history-empty? (rf/subscribe [::subs/chat-history-empty?])]
    (fn []
      (when (and (= @active-screen :chat) (not @chat-history-empty?))
        [icon-button {:on-click #(do (rf/dispatch [::chat-events/clear-chat-history])
                                     (rf/dispatch [::persistence/clear-chat-history]))
                      :size "large"
                      :color "inherit"}
         [delete-icon/delete-forever]]))))
(defn nav-bar
  "Renders the navigation bar."
  []
  [:div {:class "w-full static"}
   [app-bar {:position "static"}
    [toolbar {:variant "dense"}
     [typography {:variant "h6"} "OpenLLM"]
     [:div {:class "ml-10"}
      [button {:on-click #(rf/dispatch [:set-screen-id :playground])
               :color "inherit"
               :start-icon (r/as-element [brush-icon/brush])} "Playground"]]
     [:div {:class "ml-3 pl-3 border-l border-gray-800"}
      [button {:on-click #(rf/dispatch [:set-screen-id :chat])
               :color "inherit"
               :start-icon (r/as-element [chat-icon/chat])} "Conversation"]
      [clear-chat-history-button]]
     [collapse-side-bar-button]]]])
