(ns openllm.components.nav-bar.views
  (:require [re-frame.core :as rf]
            [openllm.events :as root-events]
            [openllm.components.nav-bar.subs :as subs]
            [openllm.components.nav-bar.events :as events]
            [openllm.components.side-bar.events :as side-bar-events]
            [openllm.components.chat.events :as chat-events]
            [openllm.api.persistence :as persistence]
            [reagent-mui.material.app-bar :refer [app-bar]]
            [reagent-mui.material.toolbar :refer [toolbar]]
            [reagent-mui.material.icon-button :refer [icon-button]]
            [reagent-mui.material.tooltip :refer [tooltip]]
            [reagent-mui.material.svg-icon :refer [svg-icon]]
            [reagent-mui.icons.delete-forever :as delete-icon]
            [reagent-mui.icons.ios-share :as share-icon]
            [reagent-mui.icons.git-hub :as github-icon]
            [reagent-mui.material.tabs :refer [tabs]]
            [reagent-mui.material.tab :refer [tab]]))

(def discord-icon-d "M19.27 5.33C17.94 4.71 16.5 4.26 15 4a.09.09 0 0 0-.07.03c-.18.33-.39.76-.53 1.09a16.09 16.09 0 0 0-4.8 0c-.14-.34-.35-.76-.54-1.09c-.01-.02-.04-.03-.07-.03c-1.5.26-2.93.71-4.27 1.33c-.01 0-.02.01-.03.02c-2.72 4.07-3.47 8.03-3.1 11.95c0 .02.01.04.03.05c1.8 1.32 3.53 2.12 5.24 2.65c.03.01.06 0 .07-.02c.4-.55.76-1.13 1.07-1.74c.02-.04 0-.08-.04-.09c-.57-.22-1.11-.48-1.64-.78c-.04-.02-.04-.08-.01-.11c.11-.08.22-.17.33-.25c.02-.02.05-.02.07-.01c3.44 1.57 7.15 1.57 10.55 0c.02-.01.05-.01.07.01c.11.09.22.17.33.26c.04.03.04.09-.01.11c-.52.31-1.07.56-1.64.78c-.04.01-.05.06-.04.09c.32.61.68 1.19 1.07 1.74c.03.01.06.02.09.01c1.72-.53 3.45-1.33 5.25-2.65c.02-.01.03-.03.03-.05c.44-4.53-.73-8.46-3.1-11.95c-.01-.01-.02-.02-.04-.02zM8.52 14.91c-1.03 0-1.89-.95-1.89-2.12s.84-2.12 1.89-2.12c1.06 0 1.9.96 1.89 2.12c0 1.17-.84 2.12-1.89 2.12zm6.97 0c-1.03 0-1.89-.95-1.89-2.12s.84-2.12 1.89-2.12c1.06 0 1.9.96 1.89 2.12c0 1.17-.83 2.12-1.89 2.12z")

(def github-url "https://github.com/bentoml/OpenLLM")

(def discord-url "https://l.bentoml.com/join-openllm-discord")

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
    [toolbar {:variant "dense"
              :style {:height "48px"}}
     [:div
      (let [screen-id @(rf/subscribe [:screen-id])]
        [tabs {:value (if (= :chat screen-id) 1 0)
               :text-color "inherit"}
         [tab {:label "Playground"
               :id "playground-tab"
               :on-click #(rf/dispatch-sync [:set-screen-id :playground])}]
         [tab {:label "Chat"
               :id "chat-tab"
               :on-click #(do (rf/dispatch-sync [:set-screen-id :chat])
                              (rf/dispatch [::chat-events/auto-scroll]))}]])]
     [:div {:class "w-full flex justify-end items-center"}
      ;;[:div {:class "mr-8"}
      ;; [context-icon-buttons]]
      [tooltip {:title github-url}
       [icon-button {:on-click #(rf/dispatch [::root-events/open-link-in-new-tab github-url])
                     :color "secondary"
                     :size "small"}
        [github-icon/git-hub]]]
      [tooltip {:title discord-url}
       [icon-button {:on-click #(rf/dispatch [::root-events/open-link-in-new-tab discord-url])
                     :color "primary"
                     :size "small"}
        [svg-icon
         [:circle {:cx 12
                   :cy 12
                   :r 12
                   :fill "#fff"}]
         [:path {:d discord-icon-d}]]]]]]]])
