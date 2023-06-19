(ns tailwind.db
  "This namespace acts as the app-db for the application. This namespace
   and the backing data structure should be the only part of the application
   that has state and is mutable.
   Other parts of the application should stay as pure as possible.
   
   Do not directly interact with this namespace! Use the 'sub'
   (subscriptions) and event namespaces for this.
   The subscription namespaces may read the respective data that is of
   interest to them.
   The event namespaces may be used to change (thus WRITE to the) app-db."
  (:require [reagent.core :as r]))

(defonce state (r/atom {:screen-id true
                        :user-dropdown? true}))