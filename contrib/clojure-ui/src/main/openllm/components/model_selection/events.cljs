(ns openllm.components.model-selection.events
  (:require [openllm.components.model-selection.db :as db]
            [openllm.events :refer [check-spec-interceptor]]
            [openllm.api.log4cljs.core :refer [log]]
            [re-frame.core :refer [reg-event-db reg-event-fx reg-cofx inject-cofx]])
  (:require-macros [openllm.build :refer [slurp]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             Coeffects              ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-cofx
 ::model-data-json-parsed
 (fn [cofx _]
   (assoc cofx
          :model-data-json-parsed
          (-> "./src/generated/models-data.json"
              (slurp ,) ;; see `openllm.build/slurp` to see how this sorcery works
              (js/JSON.parse ,)
              (js->clj , :keywordize-keys true)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;               Events               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-event-db
 ::set-model-type
 [check-spec-interceptor]
 (fn [db [_ model-type]] 
   (let [ks (partial db/key-seq :selected-model)
         all-model-ids (get-in db (db/key-seq :all-models model-type :model_id))]
     (-> db
         (assoc-in , (ks :model-type) model-type)
         (assoc-in , (ks :model-id) (first all-model-ids))))))

(reg-event-db
 ::set-model-id
 [check-spec-interceptor]
 (fn [db [_ model-id]]
   (assoc-in db (db/key-seq :selected-model :model-id) model-id)))

(reg-event-fx
 :slurp-model-data-json
 [check-spec-interceptor (inject-cofx ::model-data-json-parsed)]
 (fn [{:keys [db model-data-json-parsed]} _]
   {:db (let [all-models (get db (db/key-seq :all-models))]
          (if (or (= db/loading-text all-models) (nil? all-models))
            (assoc-in db (db/key-seq :all-models) model-data-json-parsed)
            (do (log :warn "Attempted to slurp and parse model data json, but the db already contained data:" all-models)
                db)))}))
