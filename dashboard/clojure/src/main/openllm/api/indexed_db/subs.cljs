(ns openllm.api.indexed-db.subs
  (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::indexed-db
 (fn [db]
   (:indexed-db db)))
  