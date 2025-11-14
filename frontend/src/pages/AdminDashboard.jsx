import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LogOut, Plus, Users, Shirt } from "lucide-react";
import axios from "axios";
import { toast } from "sonner";
import UsersTable from "@/components/UsersTable";
import LockersManagement from "@/components/LockersManagement";
import AddUserDialog from "@/components/AddUserDialog";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function AdminDashboard({ user, onLogout }) {
  const [users, setUsers] = useState([]);
  const [showAddUserDialog, setShowAddUserDialog] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUsers();
  }, []);

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
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 bg-white shadow-md">
            <TabsTrigger value="users" className="data-[state=active]:bg-[#278D33] data-[state=active]:text-white" data-testid="users-tab">
              <Users className="w-4 h-4 mr-2" />
              Usuarios
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
                  <UsersTable users={users} onDelete={handleDeleteUser} />
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
                <LockersManagement swimmers={users.filter(u => u.role === 'swimmer')} />
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
    </div>
  );
}
