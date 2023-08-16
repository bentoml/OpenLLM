(ns openllm.components.side-bar.model-selection.subs
    (:require [openllm.components.side-bar.model-selection.db :as db]
              [openllm.components.side-bar.subs :as side-bar-subs]
              [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::selected-model
 :<- [::side-bar-subs/model-selection-db]
 (fn [model-selection-db _]
   (:selected-model model-selection-db)))

(reg-sub
 ::all-models-data
 :<- [::side-bar-subs/model-selection-db]
 (fn [model-selection-db _]
   (:all-models model-selection-db)))

(reg-sub
 ::selected-model-type
 :<- [::selected-model]
 (fn [selected-model _]
   (:model-type selected-model)))

(reg-sub
 ::selected-model-id
 :<- [::selected-model]
 (fn [selected-model _]
   (:model-id selected-model)))

;; Returns a list of all `model-types`.
(reg-sub
 ::all-model-types
 :<- [::all-models-data]
 (fn [all-models-data _]
   (-> all-models-data
       (keys ,)
       (conj , db/loading-text))))

;; Returns a list of all `model-ids` for all `model-types`.
(reg-sub
 ::all-model-ids
 :<- [::all-models-data]
 (fn [all-models-data _]
   (conj
    (->> all-models-data
         (mapv (fn [[_ model-type]]
                 (:model_id model-type)) ,)
         (apply concat ,)
         (vec ,))
    db/loading-text)))
