(ns openllm.api.indexed-db.core
  "The indexed-db API may be accessed through this namespace. This
   namespace is meant to be used directly, this is not a db namespace
   in the re-frame sense, but an independent API which will be used for
   client-side persistence.
   false
   I recommend to import this namespace as `idb` to avoid any confusion."
  (:require [openllm.api.indexed-db.events :as events]
            [openllm.api.indexed-db.subs :as subs]
            [re-frame.core :as re-frame]
            [reagent.core :as r]))

(def db-name "OpenLLM_clj_GutZuFusss")
(def db-version 4)

(def ^:private ^:const READ_WRITE "readwrite")
(def ^:private ^:const READ_ONLY "readonly")

(def table-chat-history
  {:name "chat-history"
   :index [{:name "user" :unique false}]})

(defn idb-error-callback
  "This function is called when an error occurs during an IndexedDB
   request.
   It will log the error to the browser's console."
  [e]
  (.error js/console "Error during IndexedDB request" e))

(defn create-object-store!
  "Create an object store inside a database.
   Will return the object store object with a transaction attached."
  [obj-store-fqn table-info]
  (let [{:keys [db os-name]} obj-store-fqn
        object-store (.createObjectStore db os-name #js {:keyPath "id" :autoIncrement true})]
    (for [table-idx (:index table-info)]
      (.createIndex
       object-store
       (:name table-idx) (:name table-idx) #js {:unique (:unique table-idx)}))
    (set! (.. object-store -transaction -oncomplete)
          #(-> db
                (.transaction os-name "readwrite")
                (.objectStore os-name)))))

(defn- create-transaction
  "Create a transaction. This function is meant to be used by the
   object store ('os-*') functions. This function will create a 
   transaction and return the object store, which can be used
   right away.
   
   We consider this function semi-pure since there are no *notable* 
   direct side effects."
  [obj-store-fqn mode]
  (let [{:keys [db os-name]} obj-store-fqn
        transaction (.transaction db #js [os-name] mode)]
    (set! (.-onerror transaction) idb-error-callback)
    (-> transaction
        (.objectStore os-name))))

(defn os-add!
  "Add an object to the given object store. This function will
   create a transaction and add the object to the object store."
  [obj-store-fqn entry]
  (let [object-store (create-transaction obj-store-fqn READ_WRITE)]
    (-> object-store
        (.put (clj->js entry)))))

(defn os-add-all!
  "Add an vector of objects to the given object store. This function will
   create a transaction and add the objects to the object store.
   Note that we create a new transaction for each object.
   TODO: Figure out if there is a way with clojure looping constructs to
   create a single transaction and add all objects to the object store
   at once.
   
   An exception will be thrown, if the third argument is not a vector!
   There are no guarantees that the objects will be added in the excpected
   order if the algorithm is not adjusted, so for not no other collections
   are allowed."
  [obj-store-fqn entries]
    (when-not (vector? entries)
      (throw
       (ex-info "os-add-all! expects a vector of objects as its third argument."
                {:entries entries})))
    (loop [entries entries]
      (if (empty? entries)
        nil
        (do
          (print (first entries))
          (os-add! obj-store-fqn (first entries))
          (recur (rest entries))))))

(defn os-index->object
  "Use this function to get a single object from the object store. In
   order to retrieve all objects from the object store, use the function
   `os-get-all` instead."
  [obj-store-fqn idx callback-fn]
  (let [object-store (create-transaction obj-store-fqn READ_ONLY)
        request (.get object-store idx)]
    (set! (.-onerror request) idb-error-callback)
    (set! (.-onsuccess request)
          (fn [e]
            (callback-fn (.-result (.-target e)))))))

(defn os-get-all
  "Get all objects from the object store. This function will create
   a transaction and get all objects from the object store.
   callback-fn should be a function that takes a vector of objects
   as its only argument.
   
   It is up for discussion, whether this function should be considered
   to have side effects or not. I think it should be given some thoughts,
   because it opens a transaction and thus locks the data-base and it
   will call the callback-fn with the objects from the object store.
   It might be possible to at least get rid of the atom."
  [obj-store-fqn callback-fn]
  (let [values (r/atom []) ;; grrr
        request
        (->
         (create-transaction obj-store-fqn READ_ONLY)
         .openCursor)]
    (set! (.-onerror request) idb-error-callback)
    (set! (.-onsuccess request)
          (fn [e]
            (if-let [cursor (.. e -target -result)]
              (do
                (swap! values conj (.-value cursor))
                (.continue cursor))
              (callback-fn @values))))))

(defn- wipe-object-store!
  "Wipe the object store identified by os-name and the database. 
   This function is meant to be used for testing purposes only, it
   should stay private and only be called via the REPL. Or test
   runners... eventually..."
  [obj-store-fqn]
  (let [object-store (create-transaction obj-store-fqn READ_WRITE)
        transaction (.-transaction object-store)]
    (set! (.-oncomplete transaction) #(print "Object stores wiped."))
    (set! (.-onerror transaction) idb-error-callback)
    (.clear object-store)))

(defn on-upgrade-needed!
  "This function is called as a callback when the database is upgraded.
   It will create the object stores for the application and and save the
   backing database in the app-db for later use."
  [os-name e]
  (let [db (.. e -target -result)]
    (re-frame/dispatch-sync [::events/init-indexed-db db])
    (create-object-store! {:db db :os-name os-name}
                          table-chat-history)))

(defn initialize!
  "Initialize the indexed-db. This function should be called once
   when the application starts. This function will pull the database
   from the browser and register a callback that creates an object
   store."
  []
  (let [upgrade-callback! (partial on-upgrade-needed! (:name table-chat-history))
        request (.open
                 (. js/window -indexedDB)
                 db-name db-version)]
    (set! (.-onerror request) idb-error-callback)
    (set! (.-onupgradeneeded request) upgrade-callback!)))


(comment
  (def idb-sub (re-frame/subscribe [::subs/indexed-db])) ;; => [#object[reagent.ratom.Reaction {:val #object[IDBDatabase [object IDBDatabase]]}]]

  (def obj-store-name (:name table-chat-history)) ;; => ["chat-history"]

  (def obj-store-fqn {:db @idb-sub :os-name obj-store-name})

  (def test-messages [{:user :user :text "Hey"}
                      {:user :model :text "Hey, how are you?"}
                      {:user :user :text "I'm fine, thanks."}
                      {:user :model :text "That's good to hear."}])

  
  ;; very simple sanity check
  (. js/window -indexedDB) ;; => #object[IDBFactory [object IDBFactory]]
  

  ;; initialize the database and creates the object stores
  (initialize!) ;; => #object[openllm$api$indexed_db$core$on_upgrade_needed]

  
  ;; add the test messages from above to the object store
  (os-add-all! obj-store-fqn
               test-messages) ;; => nil


  ;; get the first object from the object store
  (os-index->object obj-store-fqn
                    1
                    #(print (js->clj % :keywordize-keys true))) ;; => #object[Function]
  ;; and prints: {:user :user, :text Hey}


  ;; get all objects from the object store
  (os-get-all obj-store-fqn
              #(print (js->clj % :keywordize-keys true))) ;; => #object[Function]
   ;; and prints: "{:user :user, :text Hey}{:user :model, :text Hey, how are you?}{:user :user, :text I'm fine, thanks.}{:user :model, :text That's good to hear.}#object[Function]"



  ;; this will wipe the object store
  (wipe-object-store! obj-store-fqn))
