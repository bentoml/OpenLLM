(ns openllm.api.indexed-db.core
  "The indexed-db API may be accessed through this namespace. This
   namespace is meant to be used directly, this is not a db namespace
   in the re-frame sense, but an independent API which will be used for
   client-side persistence.
   false
   I recommend to import this namespace as `idb` to avoid any confusion."
  (:require [openllm.api.log4cljs.core :refer [log]]))

(def ^:private ^:const READ_WRITE "readwrite")
(def ^:private ^:const READ_ONLY "readonly")

(defn idb-error-callback
  "This function is called when an error occurs during an IndexedDB
   request.
   It will log the error to the browser's console."
  [e]
  (log :error "Error during IndexedDB request" e))

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
                (.transaction os-name READ_WRITE)
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
   create a transaction and add the object to the object store.
   Returns nil."
  [obj-store-fqn entry]
  (let [object-store (create-transaction obj-store-fqn READ_WRITE)]
    (-> object-store
        (.put (clj->js entry))))
  nil)

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
        (do (os-add! obj-store-fqn (first entries))
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
            (callback-fn (.-result (.-target e))))))
  nil)

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
  (let [values (atom []) ;; TODO: grrr
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
              (callback-fn @values)))))
  nil)

(defn- wipe-object-store!
  "Wipe the object store identified by os-name and the database. 
   This function is meant to be used for testing purposes only, it
   should stay private and only be called via the REPL. Or test
   runners... eventually..."
  [obj-store-fqn]
  (let [object-store (create-transaction obj-store-fqn READ_WRITE)
        transaction (.-transaction object-store)]
    (set! (.-oncomplete transaction) #(log :info "Object store wiped."))
    (set! (.-onerror transaction) idb-error-callback)
    (.clear object-store))
  nil)

(defn- on-upgrade-needed!
  "This function is called as a callback when the database is upgraded.
   It will create the object stores for the application and and save the
   backing database in the app-db for later use.

   There are two possible reasons that the database got upgraded:
   1. The database did not exist before and was created.
   2. The database existed before, but the version (and presumably the
      schema) was lower/older than the current version."
  [table-info user-callback e]
  (let [db (.. e -target -result)
        old-version (.-oldVersion db)
        new-version (.-version db)]
    (when (and (> 0 old-version) (nil? user-callback))
      (throw
       (ex-info "The database version was upgraded, but no 'on-upgrade-db-version' callback was provided to 'initialize!'."
                {:old-version old-version :new-version new-version})))
    (if (some? user-callback)
      (do (log :info (str "Received upgrade needed event, current version is " new-version ", old version is " old-version ". Calling user callback."))
          (user-callback old-version new-version))
      (do (log :info (str "Database and object store created. Current database version is " (.-version db) "."))
          (create-object-store! {:db db :os-name (:name table-info)} table-info)))
    nil))

(defn- on-initialize-success!
  "This function is called as a callback when the database is initialized.
   It will call the callback fn passed into `initialize` by the user."
  [user-callback e]
  (let [db (.. e -target -result)]
    (user-callback db)
    (log :debug "Database initialized and callback function triggered." e)))

(defn initialize!
  "Initialize the indexed-db. This function should be called once
   when the application starts. The `db-init-callback` function will
   be called when the database is initialized. It will be called with
   the database as its only argument.

   Optionally you can pass an on-upgrade callback function, which will
   be called when the database is upgraded to a new version. The
   function will be called with the old version number as its first
   and the new version as its last argument. If you do pass a function,
   it will be called instead of the default callback function. This
   means, that you will have to create the object store.
   An example of how to do this:
   ```clojure
   (create-object-store! {:db db :os-name store-name}
                         your-table-definition)
   ```"
  ([db-info table-info db-init-callback] (initialize! db-info table-info db-init-callback nil))
  ([db-info table-info db-init-callback on-upgrade-db-version]
   (let [{:keys [db-name db-version]} db-info
         upgrade-callback! (partial on-upgrade-needed!
                                    table-info
                                    on-upgrade-db-version)
         request (.open
                  (. js/window -indexedDB)
                  db-name db-version)]
     (set! (.-onerror request) idb-error-callback)
     (set! (.-onupgradeneeded request) upgrade-callback!)
     (set! (.-onsuccess request) (partial on-initialize-success! db-init-callback)))
   nil))


;; rich comments for documentation purposes. execute in order to get the same results
;; as the ones in the ";; =>" comments
(comment
  (def db-atom (atom nil)) ;; => [#object[cljs.core.Atom {:val #object[IDBDatabase [object IDBDatabase]]}]]

  (def obj-store-name "chat-history") ;; => ["chat-history"]

  (def obj-store-fqn {:db @db-atom :os-name obj-store-name})

  (def test-messages [{:user :user :text "Hey"}
                      {:user :model :text "Hey, how are you?"}
                      {:user :user :text "I'm fine, thanks."}
                      {:user :model :text "That's good to hear."}])
  
  ;; test ultra advanced logging framework. checks all the boxes for an enterprise
  ;; grade logging framework:
  ;; 1. logs stuff
  ;; 2. does not allow RCE
  ;;    -> this technology is years ahead of the competition. looking at you, log4j.
  (log :warn "uptempo hardcore" 200 "bpm") ;; => nil


  ;; very simple sanity check
  (. js/window -indexedDB) ;; => #object[IDBFactory [object IDBFactory]]


  ;; initialize the database and creates the object stores
  (initialize! {:db-name "test-db" :db-version 1}
               {:name "chat-history"
                :index [{:name "user"
                         :unique false}]}
               #(reset! db-atom %)) ;; => nil


  ;; add a single test message to the object store
  (os-add! obj-store-fqn
           {:user :model :text "test message"}) ;; => nil


  ;; add the test messages from above to the object store
  (os-add-all! obj-store-fqn
               test-messages) ;; => nil


  ;; get the second object from the object store
  (os-index->object obj-store-fqn
                    2
                    #(print (js->clj % :keywordize-keys true))) ;; => nil
  ;; and prints: {:user :user, :text Hey}


  ;; get all objects from the object store
  (os-get-all obj-store-fqn
              #(print (js->clj % :keywordize-keys true))) ;; => nil
   ;; and prints a vector of size 5 with the test messages added above


  
  ;; this will wipe the object store
  (wipe-object-store! obj-store-fqn))
