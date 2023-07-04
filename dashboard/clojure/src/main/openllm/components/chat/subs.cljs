(ns openllm.components.chat.subs
    (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::chat-input-value
 (fn [db _]
   (:chat-input-value db)))

(reg-sub
 ::chat-history
 (fn [db _]
   (:chat-history db)))

(reg-sub
 ::chat-history-empty?
 (fn [db _]
   (empty? (:chat-history db))))
