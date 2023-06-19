(ns tailwind.views
  (:require [tailwind.events :as events]
            [tailwind.db :as db]))

(defn second-page
  []
  [:div {:class "min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8"}
   [:div {:class "sm:mx-auto sm:w-full sm:max-w-md"}
    [:h2 {:class "mt-6 text-center text-3xl font-extrabold text-gray-900"} "Sign in to your account"]
    [:p {:class "mt-2 text-center text-sm text-gray-600 max-w"} "Or "
     [:a {:href "#" :class "font-medium text-pink-600 hover:text-pink-500"} "start your 14-day free trial"]]]
   [:div {:class "mt-8 sm:mx-auto sm:w-full sm:max-w-md"}
    [:div {:class "bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10"} ;; rounded corners card
     [:h2 {:class "mt-6 text-center text-3xl font-extrabold text-gray-900"} "Sign in to your account"]]]])

(defn openllm-tag
  "The 'OpenLLM' tagline in the very top left corner of the screen."
  []
  [:div {:class "flex items-center flex-shrink-0 px-6"}
   [:div {:class "text-center text-3xl font-bold text-blue-1000"} "OpenLLM"]])

(defn selected-model-data
  "The part of the dropdown menu that is actually being displayed."
  []
  [:span {:class "flex min-w-0 items-center justify-between space-x-3"}
   [:img {:class "w-10 h-10 bg-gray-300 rounded-full flex-shrink-0"
          :src "https://images.unsplash.com/photo-1502685104226-ee32379fefbe?ixlib=rb-1.2.1&ixqx=4cZVjZZC0A&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=3&w=256&h=256&q=80" :alt ""}]
   [:span {:class "flex-1 min-w-0"}
    [:span {:class "text-gray-900 text-sm font-medium truncate"} "Jessy Schwarz"]
    [:br]
    [:span {:class "text-gray-500 text-sm truncate"} "@jessyschwarz"]]])

(defn model-dropdown-selected-item
  "The whole drop down menu where models are selected by the user. At the core there is a button
   that is used to toggle the dropdown menu."
  []
  [:div
   [:button {:type "button"
             :on-click #(events/toggle-user-dropdown)
             :class "group w-full bg-gray-100 rounded-md px-3.5 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-100 focus:ring-pink-500"
             :id "options-menu"
             :aria-expanded "false"
             :aria-haspopup "true"}
    [:span {:class "flex w-full justify-between items-center"}
     [selected-model-data]
     [:svg {:class "flex-shrink-0 h-5 w-5 text-gray-400 group-hover:text-gray-500" :xmlns "http://www.w3.org/2000/svg" :viewBox "0 0 20 20" :fill "currentColor" :aria-hidden "true"}
      [:path {:fill-rule "evenodd" :d "M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" :clip-rule "evenodd"}]]]]])

(defn model-dropdown-item-container
  "Renders if the dropdown is expanded. It contains all the models that the user can select."
  []
  (when (:user-dropdown? @db/state)
    [:div {:class "z-10 mx-3 origin-top absolute right-0 left-0 mt-1 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 divide-y divide-gray-200 focus:outline-none" :role "menu" :aria-orientation "vertical" :aria-labelledby "options-menu"}
     [:div {:class "py-1" :role "none"}
      [:a {:href "#" :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"} "View profile"]
      [:a {:href "#" :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"} "Settings"]
      [:a {:href "#" :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"} "Notifications"]]
     [:div {:class "py-1" :role "none"}
      [:a {:href "#"
           :on-click #(events/logout)
           :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"}
       "Logout"]]]))


(defn model-dropdown
  "Aggregates the two components making up the dropdown menu."
  []
  [:div {:class "px-3 mt-6 relative inline-block text-left"}
   [model-dropdown-selected-item]
   [model-dropdown-item-container]])

(defn side-bar
  "The render function of the toolbar on the very left of the screen"
  []
  [:div {:class "flex flex-col w-64 border-r border-gray-200 pt-5 pb-4 bg-gray-100 bg-green-200"} ;; sidebar div + background 
   [openllm-tag]
   [:div {:class "h-0 flex-1 flex flex-col overflow-y-auto"}
    [model-dropdown]
    [:nav {:class "px-3 mt-6"}
     [:div {:class "space-y-1"}
      [:a {:href "#" :class "bg-gray-200 text-gray-900 group flex items-center px-2 py-2 text-sm font-medium rounded-md"}
       [:svg {:class "text-gray-500 mr-3 h-6 w-6" :xmlns "http://www.w3.org/2000/svg" :fill "none" :viewBox "0 0 24 24" :stroke "currentColor" :aria-hidden "true"}
        [:path {:stroke-linecap "round" :stroke-linejoin "round" :stroke-width "2" :d "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"}]] "Home"]
      [:a {:href "#" :class "text-gray-700 hover:text-gray-900 hover:bg-gray-50 group flex items-center px-2 py-2 text-sm font-medium rounded-md"}
       [:svg {:class "text-gray-400 group-hover:text-gray-500 mr-3 h-6 w-6" :xmlns "http://www.w3.org/2000/svg" :fill "none" :viewBox "0 0 24 24" :stroke "currentColor" :aria-hidden "true"}
        [:path {:stroke-linecap "round" :stroke-linejoin "round" :stroke-width "2" :d "M4 6h16M4 10h16M4 14h16M4 18h16"}]] "My tasks"]]
     [:div {:class "mt-8"}
      [:h3 {:class "px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider" :id "teams-headline"} "Teams"]
      [:div {:class "mt-1 space-y-1" :role "group" :aria-labelledby "teams-headline"}
       [:a {:href "#" :class "group flex items-center px-3 py-2 text-sm font-medium text-gray-700 rounded-md hover:text-gray-900 hover:bg-gray-50"}
        [:span {:class "w-2.5 h-2.5 mr-4 bg-pink-500 rounded-full" :aria-hidden "true"}]
        [:span {:class "truncate"} "Engineering"]]
       [:a {:href "#" :class "group flex items-center px-3 py-2 text-sm font-medium text-gray-700 rounded-md hover:text-gray-900 hover:bg-gray-50"}
        [:span {:class "w-2.5 h-2.5 mr-4 bg-green-500 rounded-full" :aria-hidden "true"}]
        [:span {:class "truncate"} "Human Resources"]]]]]]])

(defn dashboard
  []
  [:div {:class "h-screen flex overflow-hidden bg-white"}
   [:div {:class "hidden lg:flex lg:flex-shrink-0"}
    [side-bar]]
   [:div {:class "flex flex-col w-0 flex-1 overflow-hidden"}
    [:main {:class "flex-1 relative z-0 overflow-y-auto focus:outline-none" :tabIndex "0"}
     [:div {:class "px-4 mt-6 sm:px-6 lg:px-8"}
      [:h2 {:class "text-gray-500 text-xs font-medium uppercase tracking-wide"} "Main"]]]]])
