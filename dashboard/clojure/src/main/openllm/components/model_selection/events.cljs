(ns openllm.components.model-selection.events
  (:require [re-frame.core :refer [reg-event-db]]))

(reg-event-db
 ::set-model-type
 (fn [db [_ model-type]]
   (assoc-in db [:selected-model :model-type] model-type)))

(reg-event-db
 ::set-model-id
 (fn [db [_ model-id]]
   (assoc-in db [:selected-model :model-id] model-id)))
