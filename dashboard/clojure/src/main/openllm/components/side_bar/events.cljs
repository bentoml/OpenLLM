(ns openllm.components.side-bar.events
    (:require [re-frame.core :refer [reg-event-db]]
              [openllm.events :refer [check-spec-interceptor]]))

(reg-event-db
 ::set-model-config-parameter
 [check-spec-interceptor]
 (fn [db [_ parameter value]]
   (assoc-in db [:model-config parameter] value)))
