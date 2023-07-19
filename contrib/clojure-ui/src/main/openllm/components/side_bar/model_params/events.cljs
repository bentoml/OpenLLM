(ns openllm.components.side-bar.model-params.events
  (:require [openllm.components.side-bar.model-params.db :as db]
            [re-frame.core :refer [reg-event-db]]
            [openllm.events :refer [check-spec-interceptor]]))

(reg-event-db
 ::set-model-config-parameter
 [check-spec-interceptor]
 (fn [db [_ parameter value]]
   (assoc-in db (db/key-seq parameter) value)))
