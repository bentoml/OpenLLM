(ns tailwind.events
    (:require [tailwind.db :as db]))

(defn login
    []
    (swap! db/state assoc :screen-id true))

(defn logout
    []
    (swap! db/state assoc :screen-id false))

(defn toggle-user-dropdown
    []
    (let [dropdown (:user-dropdown? @db/state)]
        (swap! db/state assoc :user-dropdown? (not dropdown))))