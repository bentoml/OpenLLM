(ns openllm.components.nav-bar.subs
  (:require [openllm.components.chat.subs :as chat-subs]
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
