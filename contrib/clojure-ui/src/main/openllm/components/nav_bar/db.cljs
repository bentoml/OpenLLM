(ns openllm.components.nav-bar.db
  "The branch of the `app-db` that saves data related to the `nav-bar`. This
   should have very few keys, since the `nav-bar` mostly affects other components.
   The path to this branch can be expressed as:
   *root -> components -> nav-bar*"
  (:require [cljs.spec.alpha :as s]))

(defn key-seq
  [& more-keys]
  (into [:components-db :chat-db] more-keys))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::nav-bar-db #(and (map? %) (empty? %)))

(defn initial-db
  []
  {})

(comment
  ;; check if initial-db is valid
  (s/valid? ::nav-bar-db (initial-db))
  
  ;; check if the map is allowed to have keys
  (s/valid? ::nav-bar-db {:foo "bar"}))
