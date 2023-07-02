(ns openllm.chat.subs
    (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::chat-input-value
 (fn [db _]
   (:chat-input-value db)))

(reg-sub
 ::chat-history
 (fn [db _]
   (:chat-history db)))
