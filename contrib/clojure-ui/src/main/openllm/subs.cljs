(ns openllm.subs
  (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 :screen-id
 (fn [db _]
   (:screen-id db)))

(reg-sub
 ::components-db
 (fn [db _]
   (:components-db db)))
