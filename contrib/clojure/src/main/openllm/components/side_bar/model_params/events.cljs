(ns openllm.components.side-bar.model-params.events
  (:require [openllm.components.side-bar.model-params.db :as db]
            [re-frame.core :refer [reg-event-db]]
            [openllm.events :refer [check-spec-interceptor]]))

(reg-event-db
 ::set-model-config-parameter
 [check-spec-interceptor]
 (fn [db [_ parameter value]]
   (let [type-pred (get-in db/parameter-meta-data [parameter :type-pred])
         parsed-value (condp = type-pred ;; this can probably be rewritten smarter... TODO i guess...
                        boolean? (if (boolean? value) value (parse-boolean value))
                        int? (if (int? value) value (parse-long value))
                        float? (if (float? value) value (parse-double value))
                        value)] ;; best effort probably was not enough ;_;
     (assoc-in db (db/key-seq parameter) parsed-value))))
