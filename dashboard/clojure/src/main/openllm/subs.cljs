(ns openllm.subs
  (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 :screen-id
 (fn [db _]
   (:screen-id db)))

(reg-sub
 ::model-config
 (fn [db _]
   (:model-config db)))
