(ns openllm.components.db
  "The branch of the app-db that saves data related to the chat view. Components
   are an abstract concept, this namespace is used to group all components' db
   branches; I do not think that there will be any actual fields in this namespace.
   The path to this branch can be expressed as:
   *root -> components*"
  (:require [openllm.components.chat.db :as chat-db]
            [openllm.components.nav-bar.db :as nav-bar-db]
            [openllm.components.playground.db :as playground-db]
            [openllm.components.side-bar.db :as side-bar-db]
            [clojure.spec.alpha :as s]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::components-db (s/keys :req-un [::chat-db/chat-db
                                        ::nav-bar-db/nav-bar-db
                                        ::playground-db/playground-db
                                        ::side-bar-db/side-bar-db]))

(defn initial-db
  "Initial values for this branch of the app-db."
  []
  {:chat-db (chat-db/initial-db)
   :nav-bar-db (nav-bar-db/initial-db)
   :playground-db (playground-db/initial-db)
   :side-bar-db (side-bar-db/initial-db)})

(comment
  ;; check if initial-db is valid
  (s/valid? ::components-db (initial-db)))
