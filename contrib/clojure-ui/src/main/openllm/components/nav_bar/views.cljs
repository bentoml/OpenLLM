(ns openllm.components.nav-bar.views
  (:require [re-frame.core :as rf]
            [openllm.events :as root-events]
            [openllm.components.nav-bar.subs :as subs]
            [openllm.components.nav-bar.events :as events]
            [openllm.components.side-bar.events :as side-bar-events]
            [openllm.components.chat.events :as chat-events]
            [openllm.api.persistence :as persistence]
            [reagent-mui.icons.chat :as chat-icon]
            [reagent-mui.icons.brush :as brush-icon]
            [reagent-mui.material.app-bar :refer [app-bar]]
            [reagent-mui.material.toolbar :refer [toolbar]]
            [reagent-mui.material.button :refer [button]]
            [reagent-mui.material.icon-button :refer [icon-button]]
            [reagent-mui.material.typography :refer [typography]]
            [reagent-mui.material.tooltip :refer [tooltip]] 
            [reagent-mui.icons.delete-forever :as delete-icon]
            [reagent-mui.icons.ios-share :as share-icon]
            [reagent-mui.icons.git-hub :as github-icon]
            [reagent.core :as r]))

(defn- collapse-side-bar-button
  "The collapse side bar button. It changes its icon depending on whether
   the side bar is collapsed or not."
  []
  (let [tooltip-text-collapse-sidebar (rf/subscribe [::subs/tooltip-text-collapse-sidebar])
        collapse-icon (rf/subscribe [::subs/collapse-icon])]
    (fn []
      [tooltip {:title @tooltip-text-collapse-sidebar}
       [icon-button {:on-click #(rf/dispatch [::side-bar-events/toggle-side-bar])
                     :color "inherit"}
        @collapse-icon]])))

(defn- context-icon-buttons
  "Displays the icon buttons on the very right of the navigation bar. Some
   of them are only displayed conditionally (e.g. if a certain screen is
   active)."
  []
  (let [active-screen (rf/subscribe [:screen-id])
        chat-history-empty? (rf/subscribe [::subs/chat-history-empty?])
        tooltip-text-export (rf/subscribe [::subs/tooltip-text-export])]
    (fn []
      [:<> 
       [tooltip {:title @tooltip-text-export}
        [icon-button {:on-click #(rf/dispatch [::events/export-button-clicked])
                      :size "large"
                      :color "inherit"}
         [share-icon/ios-share]]]
       (when (and (= @active-screen :chat) (not @chat-history-empty?))
         [tooltip {:title "Clear chat history"}
          [icon-button {:on-click #(do (rf/dispatch [::chat-events/clear-chat-history])
                                       (rf/dispatch [::persistence/clear-chat-history]))
                        :size "large"
                        :color "inherit"}
           [delete-icon/delete-forever]]])])))

(defn nav-bar
  "Renders the navigation bar. The navigation bar is always visible and contains the
   navigation buttons and the context dependent icon buttons. There are also small
   buttons that will open socials in a new tab."
  []
  [:div {:class "w-full static"}
   [app-bar {:position "static"
             :color "primary"}
    [toolbar {:variant "dense"}
     [icon-button {:on-click #(rf/dispatch [::root-events/open-link-in-new-tab "https://github.com/bentoml/OpenLLM"])
                   :color "inherit"}
      [github-icon/git-hub]]
     [typography {:variant "h6"} "OpenLLM"]
     [:div {:class "ml-10"}
      [button {:on-click #(rf/dispatch [:set-screen-id :playground])
               :color "inherit"
               :start-icon (r/as-element [brush-icon/brush])} "Playground"]]
     [:div {:class "ml-3 pl-3 border-l border-gray-800"}
      [button {:on-click #(do (rf/dispatch-sync [:set-screen-id :chat])
                              (rf/dispatch [::chat-events/auto-scroll]))
               :color "inherit"
               :start-icon (r/as-element [chat-icon/chat])} "Conversation"]]
     [:div {:class "w-full flex justify-end items-center"}
      [:div {:class "mr-8"}
       [context-icon-buttons]]
      [:div {:class "-mr-8"}
       [collapse-side-bar-button]]]]]])
