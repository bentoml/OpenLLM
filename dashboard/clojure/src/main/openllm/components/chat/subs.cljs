(ns openllm.components.chat.subs
    (:require [openllm.subs :as root-subs]
              [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::chat-input-value
 (fn [db _]
   (:chat-input-value db)))

(reg-sub
 ::chat-history
 (fn [db _]
   (:chat-history db)))

(reg-sub
 ::modal-open?
 :<- [:modal-open?-map]
 (fn [modal-open?-map _]
   (get modal-open?-map :chat)))

(reg-sub
 ::prompt-layout
 (fn [db _]
   (:prompt-layout db)))
