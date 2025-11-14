import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LogOut, Plus, Users, Shirt, Clock } from "lucide-react";
import axios from "axios";
import { toast } from "sonner";
import UsersTable from "@/components/UsersTable";
import LockersManagement from "@/components/LockersManagement";
import AddUserDialog from "@/components/AddUserDialog";
import EditUserDialog from "@/components/EditUserDialog";
import SwimTimesTable from "@/components/SwimTimesTable";
import AddTimeDialog from "@/components/AddTimeDialog";
import EditTimeDialog from "@/components/EditTimeDialog";
import SwimmerSelector from "@/components/SwimmerSelector";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function AdminDashboard({ user, onLogout }) {
  const [users, setUsers] = useState([]);
  const [times, setTimes] = useState([]);
  const [selectedSwimmer, setSelectedSwimmer] = useState(null);
  const [showAddUserDialog, setShowAddUserDialog] = useState(false);
  const [showEditUserDialog, setShowEditUserDialog] = useState(false);
  const [showAddTimeDialog, setShowAddTimeDialog] = useState(false);
  const [showEditTimeDialog, setShowEditTimeDialog] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [editingTime, setEditingTime] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUsers();
    fetchAllTimes();
  }, []);

  useEffect(() => {
    if (selectedSwimmer) {
      fetchTimesForSwimmer(selectedSwimmer);
    } else {
      fetchAllTimes();
    }
  }, [selectedSwimmer]);

  const fetchUsers = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      const response = await axios.get(`${API}/users`, { headers });
      setUsers(response.data);
    } catch (error) {
      toast.error("Error al cargar usuarios");
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

  const handleAddUser = async (userData) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.post(`${API}/auth/register`, userData, { headers });
      toast.success("Usuario creado correctamente");
      setShowAddUserDialog(false);
      fetchUsers();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Error al crear usuario");
    }
  };

  const handleEditUser = (user) => {
    setEditingUser(user);
    setShowEditUserDialog(true);
  };

  const handleUpdateUser = async (userData) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.put(`${API}/users/${editingUser.id}`, userData, { headers });
      toast.success("Usuario actualizado correctamente");
      setShowEditUserDialog(false);
      setEditingUser(null);
      fetchUsers();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Error al actualizar usuario");
    }
  };

  const handleDeleteUser = async (userId) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.delete(`${API}/users/${userId}`, { headers });
      toast.success("Usuario eliminado");
      fetchUsers();
    } catch (error) {
      toast.error("Error al eliminar usuario");
    }
  };

  const handleAddTime = async (timeData) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.post(`${API}/times`, timeData, { headers });
      toast.success("Tiempo registrado correctamente");
      setShowAddTimeDialog(false);
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
    setShowEditTimeDialog(true);
  };

  const handleUpdateTime = async (timeData) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      await axios.put(`${API}/times/${editingTime.id}`, timeData, { headers });
      toast.success("Tiempo actualizado correctamente");
      setShowEditTimeDialog(false);
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

  const swimmers = users.filter(u => u.role === 'swimmer');

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold text-[#278D33]">Panel de Administración</h1>
              <p className="text-gray-600 mt-1">Gestión completa del club - {user.name}</p>
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
        <Tabs defaultValue="users" className="space-y-6">
          <TabsList className="grid w-full max-w-3xl mx-auto grid-cols-3 bg-white shadow-md">
            <TabsTrigger value="users" className="data-[state=active]:bg-[#278D33] data-[state=active]:text-white" data-testid="users-tab">
              <Users className="w-4 h-4 mr-2" />
              Usuarios
            </TabsTrigger>
            <TabsTrigger value="times" className="data-[state=active]:bg-[#278D33] data-[state=active]:text-white" data-testid="times-tab">
              <Clock className="w-4 h-4 mr-2" />
              Tiempos
            </TabsTrigger>
            <TabsTrigger value="lockers" className="data-[state=active]:bg-[#278D33] data-[state=active]:text-white" data-testid="lockers-tab">
              <Shirt className="w-4 h-4 mr-2" />
              Taquillas
            </TabsTrigger>
          </TabsList>

          <TabsContent value="users" className="fade-in">
            <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <div className="flex justify-between items-center">
                  <CardTitle className="text-2xl text-gray-900">Gestión de Usuarios</CardTitle>
                  <Button
                    onClick={() => setShowAddUserDialog(true)}
                    className="bg-[#278D33] hover:bg-[#1f6b28] text-white"
                    data-testid="add-user-button"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Nuevo Usuario
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-center py-8">Cargando...</div>
                ) : (
                  <UsersTable users={users} onDelete={handleDeleteUser} onEdit={handleEditUser} />
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="times" className="fade-in">
            <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <CardTitle className="text-2xl text-gray-900">Gestión de Tiempos</CardTitle>
                  <div className="flex gap-3 w-full sm:w-auto">
                    <SwimmerSelector
                      swimmers={swimmers}
                      selectedSwimmer={selectedSwimmer}
                      onSelectSwimmer={setSelectedSwimmer}
                    />
                    <Button
                      onClick={() => setShowAddTimeDialog(true)}
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
          </TabsContent>

          <TabsContent value="lockers" className="fade-in">
            <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl text-gray-900">Gestión de Taquillas</CardTitle>
              </CardHeader>
              <CardContent>
                <LockersManagement swimmers={swimmers} />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      <AddUserDialog
        open={showAddUserDialog}
        onOpenChange={setShowAddUserDialog}
        onSubmit={handleAddUser}
      />

      <EditUserDialog
        open={showEditUserDialog}
        onOpenChange={setShowEditUserDialog}
        onSubmit={handleUpdateUser}
        user={editingUser}
      />

      <AddTimeDialog
        open={showAddTimeDialog}
        onOpenChange={setShowAddTimeDialog}
        onSubmit={handleAddTime}
        swimmers={swimmers}
      />

      <EditTimeDialog
        open={showEditTimeDialog}
        onOpenChange={setShowEditTimeDialog}
        onSubmit={handleUpdateTime}
        time={editingTime}
        swimmers={swimmers}
      />
    </div>
  );
}
