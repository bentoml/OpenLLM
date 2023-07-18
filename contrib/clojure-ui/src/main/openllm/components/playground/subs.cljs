(ns openllm.components.playground.subs
    (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::playground-input-value
 (fn [db _]
   (:playground-input-value db)))

(reg-sub
 ::last-response
 (fn [db _]
   (:playground-last-response db)))

(reg-sub
 ::modal-open?
 :<- [:modal-open?-map]
 (fn [map _]
   (:playground (or map {}))))
