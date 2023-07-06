(ns openllm.app
    (:require [openllm.api.persistence :as persistence]
              [reagent.dom :as dom]
              [openllm.views :as views]
              [re-frame.core :as rf]

              ;; the following are only required to make the compiler load the namespaces
              [day8.re-frame.http-fx]
              [openllm.events]
              [openllm.subs]
              [openllm.api.http]))

(defn app
  "The main app component, which is rendered into the DOM. This component
   just wraps the dashboard component, which is the effective root
   component of the application."
  []
  [views/dashboard])

(defn ^:dev/after-load start
  "Starts the app by rendering the app component into the DOM. This
   function is the root rendering function, and is called by the
   `init` function right after the databases are initialized."
  []
  (dom/render [app]
              (.getElementById js/document "app")))
  (let [root (rdom/create-root (js/document.getElementById "app"))]
    (rdom/render root [app])))

(defn init
  "This init function is called exactly once when the page loads.
   Responsible for initializing the app-db as well as the IndexedDB
   (persistent) database.

   This marks the entry point of the application, and is called by shadow-cljs
   directly."
  []
  (enable-console-print!) ;; so that print writes to `console.log`
  (rf/dispatch-sync [:initialise-db])
  (persistence/init-idb)
  (start))
