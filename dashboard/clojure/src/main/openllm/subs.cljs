(ns openllm.subs
  (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 :screen-id
 (fn [db _]
   (:screen-id db)))

(reg-sub
 :chat-input-value
 (fn [db _]
   (:chat-input-value db)))

(reg-sub
 :chat-history
 (fn [db _]
   (:chat-history db)))

(reg-sub
 :model-config
 (fn [db _]
   (:model-config db)))
