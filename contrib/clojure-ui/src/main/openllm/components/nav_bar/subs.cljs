(ns openllm.components.nav-bar.subs
  (:require [openllm.components.chat.subs :as chat-subs]
            [openllm.components.side-bar.subs :as side-bar-subs]
            [reagent-mui.icons.keyboard-double-arrow-right :as right-icon]
            [reagent-mui.icons.keyboard-double-arrow-left :as left-icon]
            [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::chat-history-empty?
 :<- [::chat-subs/chat-history]
 (fn [chat-history _]
   (empty? chat-history)))

(reg-sub
 ::tooltip-text-export
 :<- [:screen-id]
 (fn [screen-id _]
   (str "Export " (if (= screen-id :playground) "playground data" "chat history"))))

(reg-sub
 ::tooltip-text-collapse-sidebar
 :<- [::side-bar-subs/side-bar-open?]
 (fn [side-bar-open? _]
   (str (if side-bar-open? "Collapse" "Expand") " side bar")))

(reg-sub
 ::collapse-icon
 :<- [::side-bar-subs/side-bar-open?]
 (fn [side-bar-open? _]
   (if side-bar-open?
     [right-icon/keyboard-double-arrow-right]
     [left-icon/keyboard-double-arrow-left])))
