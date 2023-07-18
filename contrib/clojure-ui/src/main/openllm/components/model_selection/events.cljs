(ns openllm.components.model-selection.events
  (:require [openllm.components.model-selection.data :as data]
            [re-frame.core :refer [reg-event-db]]))

(reg-event-db
 ::set-model-type
 (fn [db [_ model-type]] 
   (-> db
       (assoc-in , [:selected-model :model-type] model-type)
       (assoc-in , [:selected-model :model-id] (-> model-type
                                                   (data/model-ids ,) 
                                                   (first ,))))))

(reg-event-db
 ::set-model-id
 (fn [db [_ model-id]]
   (assoc-in db [:selected-model :model-id] model-id)))
