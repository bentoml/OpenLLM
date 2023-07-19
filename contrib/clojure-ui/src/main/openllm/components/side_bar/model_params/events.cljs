(ns openllm.components.side-bar.model-params.events
  (:require [openllm.components.side-bar.model-params.db :as db]
            [re-frame.core :refer [reg-event-db]]
            [openllm.events :refer [check-spec-interceptor]]))

;; will receive the value as string
(reg-event-db
 ::set-model-config-parameter
 [check-spec-interceptor]
 (fn [db [_ parameter value]]
   (let [type-pred (get-in db/parameter-meta-data [parameter :type-pred])
         parsed-value (if (= type-pred float?)
                        (parse-double value)
                        (if (= type-pred int?)
                          (parse-long value)
                          value))]
     (assoc-in db (db/key-seq parameter) parsed-value))))
