(ns openllm.subs
  (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 :screen-id
 (fn [db _]
   (:screen-id db)))

;; intended to be used as a parent subscription, not direct use
(reg-sub
 :modal-open?-map
 (fn [db _]
   (:modal-open? db)))

(reg-sub
 ::model-config
 (fn [db _]
   (:model-config db)))
