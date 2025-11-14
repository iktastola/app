import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LogOut, Trophy, Clock, Shirt } from "lucide-react";
import axios from "axios";
import { toast } from "sonner";
import SwimTimesTable from "@/components/SwimTimesTable";
import PersonalBestsTable from "@/components/PersonalBestsTable";
import LockerView from "@/components/LockerView";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function SwimmerDashboard({ user, onLogout }) {
  const [times, setTimes] = useState([]);
  const [personalBests, setPersonalBests] = useState([]);
  const [locker, setLocker] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };

      const [timesRes, pbRes, lockerRes] = await Promise.all([
        axios.get(`${API}/times`, { headers }),
        axios.get(`${API}/personal-bests`, { headers }),
        axios.get(`${API}/lockers/${user.id}`, { headers }).catch(() => ({ data: null }))
      ]);

      setTimes(timesRes.data);
      setPersonalBests(pbRes.data);
      setLocker(lockerRes.data);
    } catch (error) {
      toast.error("Error al cargar datos");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold text-[#278D33]">Club de Natación Astolai</h1>
              <p className="text-gray-600 mt-1">Bienvenido, {user.name}</p>
            </div>
            <Button
              onClick={onLogout}
              variant="outline"
              className="border-[#278D33] text-[#278D33] hover:bg-[#278D33] hover:text-white"
              data-testid="logout-button"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Salir
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="times" className="space-y-6">
          <TabsList className="grid w-full max-w-2xl mx-auto grid-cols-3 bg-white shadow-md">
            <TabsTrigger value="times" className="data-[state=active]:bg-[#278D33] data-[state=active]:text-white" data-testid="times-tab">
              <Clock className="w-4 h-4 mr-2" />
              Mis Tiempos
            </TabsTrigger>
            <TabsTrigger value="bests" className="data-[state=active]:bg-[#278D33] data-[state=active]:text-white" data-testid="bests-tab">
              <Trophy className="w-4 h-4 mr-2" />
              Mejores Marcas
            </TabsTrigger>
            <TabsTrigger value="locker" className="data-[state=active]:bg-[#278D33] data-[state=active]:text-white" data-testid="locker-tab">
              <Shirt className="w-4 h-4 mr-2" />
              Mi Taquilla
            </TabsTrigger>
          </TabsList>

          <TabsContent value="times" className="fade-in">
            <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl text-gray-900">Mis Tiempos de Natación</CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">Cargando...</div>
                ) : (
                  <SwimTimesTable times={times} />
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="bests" className="fade-in">
            <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl text-gray-900 flex items-center gap-2">
                  <Trophy className="w-6 h-6 text-[#278D33]" />
                  Mis Mejores Marcas
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">Cargando...</div>
                ) : (
                  <PersonalBestsTable personalBests={personalBests} />
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="locker" className="fade-in">
            <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl text-gray-900">Mi Taquilla Virtual</CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">Cargando...</div>
                ) : (
                  <LockerView locker={locker} />
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
