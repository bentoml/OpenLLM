(ns openllm.components.side-bar.events
    (:require [openllm.components.side-bar.db :as db]
              [re-frame.core :refer [reg-event-db]]
              [openllm.events :refer [check-spec-interceptor]]))

(reg-event-db
 ::toggle-side-bar
 [check-spec-interceptor]
 (fn [db _]
   (let [new-value (not (get-in db (db/key-seq :side-bar-open?)))]
     (assoc-in db (db/key-seq :side-bar-open?) new-value))))

(reg-event-db
 ::set-model-config-parameter
 [check-spec-interceptor]
 (fn [db [_ parameter value]]
   (assoc-in db (db/key-seq :model-config parameter) value)))
