import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LogOut, Plus, Users } from "lucide-react";
import axios from "axios";
import { toast } from "sonner";
import SwimTimesTable from "@/components/SwimTimesTable";
import AddTimeDialog from "@/components/AddTimeDialog";
import EditTimeDialog from "@/components/EditTimeDialog";
import SwimmerSelector from "@/components/SwimmerSelector";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function CoachDashboard({ user, onLogout }) {
  const [times, setTimes] = useState([]);
  const [swimmers, setSwimmers] = useState([]);
  const [selectedSwimmer, setSelectedSwimmer] = useState(null);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [editingTime, setEditingTime] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (selectedSwimmer) {
      fetchTimesForSwimmer(selectedSwimmer);
    } else {
      fetchAllTimes();
    }
  }, [selectedSwimmer]);

  const fetchData = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };

      const [usersRes, timesRes] = await Promise.all([
        axios.get(`${API}/users`, { headers }),
        axios.get(`${API}/times`, { headers })
      ]);

      const swimmersList = usersRes.data.filter(u => u.role === 'swimmer');
      setSwimmers(swimmersList);
      setTimes(timesRes.data);
    } catch (error) {
      toast.error("Error al cargar datos");
    } finally {
      setLoading(false);
    }
  };

  const fetchAllTimes = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      const response = await axios.get(`${API}/times`, { headers });
      setTimes(response.data);
    } catch (error) {
      toast.error("Error al cargar tiempos");
    }
  };

  const fetchTimesForSwimmer = async (swimmerId) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      const response = await axios.get(`${API}/times?swimmer_id=${swimmerId}`, { headers });
      setTimes(response.data);
    } catch (error) {
      toast.error("Error al cargar tiempos");
    }
  };

  const handleAddTime = async (timeData) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.post(`${API}/times`, timeData, { headers });
      toast.success("Tiempo registrado correctamente");
      setShowAddDialog(false);
      if (selectedSwimmer) {
        fetchTimesForSwimmer(selectedSwimmer);
      } else {
        fetchAllTimes();
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || "Error al registrar tiempo");
    }
  };

  const handleEditTime = (time) => {
    setEditingTime(time);
    setShowEditDialog(true);
  };

  const handleUpdateTime = async (timeData) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.put(`${API}/times/${editingTime.id}`, timeData, { headers });
      toast.success("Tiempo actualizado correctamente");
      setShowEditDialog(false);
      setEditingTime(null);
      if (selectedSwimmer) {
        fetchTimesForSwimmer(selectedSwimmer);
      } else {
        fetchAllTimes();
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || "Error al actualizar tiempo");
    }
  };

  const handleDeleteTime = async (timeId) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.delete(`${API}/times/${timeId}`, { headers });
      toast.success("Tiempo eliminado");
      if (selectedSwimmer) {
        fetchTimesForSwimmer(selectedSwimmer);
      } else {
        fetchAllTimes();
      }
    } catch (error) {
      toast.error("Error al eliminar tiempo");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold text-[#278D33]">Panel de Entrenador</h1>
              <p className="text-gray-600 mt-1">Gestión de nadadores - {user.name}</p>
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
        <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <CardTitle className="text-2xl text-gray-900">Tiempos de Natación</CardTitle>
              <div className="flex gap-3 w-full sm:w-auto">
                <SwimmerSelector
                  swimmers={swimmers}
                  selectedSwimmer={selectedSwimmer}
                  onSelectSwimmer={setSelectedSwimmer}
                />
                <Button
                  onClick={() => setShowAddDialog(true)}
                  className="bg-[#278D33] hover:bg-[#1f6b28] text-white"
                  data-testid="add-time-button"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Registrar Tiempo
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8">Cargando...</div>
            ) : (
              <SwimTimesTable 
                times={times} 
                swimmers={swimmers}
                onDelete={handleDeleteTime}
                onEdit={handleEditTime}
                showActions
              />
            )}
          </CardContent>
        </Card>
      </main>

      <AddTimeDialog
        open={showAddDialog}
        onOpenChange={setShowAddDialog}
        onSubmit={handleAddTime}
        swimmers={swimmers}
      />

      <EditTimeDialog
        open={showEditDialog}
        onOpenChange={setShowEditDialog}
        onSubmit={handleUpdateTime}
        time={editingTime}
        swimmers={swimmers}
      />
    </div>
  );
}
