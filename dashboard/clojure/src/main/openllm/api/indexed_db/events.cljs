(ns openllm.api.indexed-db.events
  (:require [re-frame.core :as re-frame]))
  
(re-frame/reg-event-db
 ::init-indexed-db
 (fn [db [_ indexed-db]]
   (assoc db :indexed-db indexed-db)))
