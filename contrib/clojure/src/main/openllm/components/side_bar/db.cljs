(ns openllm.components.side-bar.db
  "The branch of the `app-db` that saves data related to the `side-bar` view. This
   mostly revolves around the model parameters.
   The path to this branch can be expressed as:
   *root -> components -> side-bar*"
   (:require [openllm.components.side-bar.model-selection.db :as model-selection-db]
             [openllm.components.side-bar.model-params.db :as model-params-db]
             [cljs.spec.alpha :as s]))

(defn key-seq
  "Returns the key sequence to access the side-bar-db This is useful for
   `assoc-in` and `get-in`. The `more-keys` argument is optional and can be
   used to access a sub-key of the side-bar-db
   Returns the key sequence to access the side-bar-db"
  [& more-keys]
  (into [:components-db :side-bar-db] more-keys))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::side-bar-open? boolean?)

(s/def ::side-bar-db (s/keys :req-un [::side-bar-open?
                                      ::model-selection-db/model-selection-db
                                      ::model-params-db/model-params-db]))

(defn initial-db
  "Initial values for this branch of the app-db."
  []
  {:side-bar-open? true
   :model-selection-db (model-selection-db/initial-db)
   :model-params-db (model-params-db/initial-db)})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; check if initial-db is valid
  (s/valid? ::side-bar-db (initial-db)))
