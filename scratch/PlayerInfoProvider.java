package cuspymd.mcp.mod.utils;

import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.level.Level;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.food.FoodData;
import net.minecraft.client.multiplayer.MultiPlayerGameMode;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.ItemStack;

public class PlayerInfoProvider implements IPlayerInfoProvider {

    public PlayerInfoProvider() {
    }

    @Override
    public JsonObject getPlayerInfo() {
        return getPlayerInfoStatic();
    }

    public static JsonObject getPlayerInfoStatic() {
        Minecraft client = Minecraft.getInstance();
        LocalPlayer player = client.player;
        Level world = client.level;

        JsonObject playerInfo = new JsonObject();
        if (player == null) {
            playerInfo.addProperty("error", "No player found");
            return playerInfo;
        }

        JsonObject position = new JsonObject();
        position.addProperty("x", Double.valueOf(player.getX()));
        position.addProperty("y", Double.valueOf(player.getY()));
        position.addProperty("z", Double.valueOf(player.getZ()));
        playerInfo.add("position", position);

        JsonObject blockPos = new JsonObject();
        blockPos.addProperty("x", Integer.valueOf(player.getBlockX()));
        blockPos.addProperty("y", Integer.valueOf(player.getBlockY()));
        blockPos.addProperty("z", Integer.valueOf(player.getBlockZ()));
        playerInfo.add("blockPosition", blockPos);

        JsonObject rotation = new JsonObject();
        rotation.addProperty("yaw", Float.valueOf(player.getYRot()));
        rotation.addProperty("pitch", Float.valueOf(player.getXRot()));
        playerInfo.add("rotation", rotation);

        String direction = getCardinalDirection(player.getYRot());
        playerInfo.addProperty("facingDirection", direction);

        Vec3 lookVec = player.getViewVector(1.0f);
        JsonObject lookVector = new JsonObject();
        lookVector.addProperty("x", Double.valueOf(lookVec.x));
        lookVector.addProperty("y", Double.valueOf(lookVec.y));
        lookVector.addProperty("z", Double.valueOf(lookVec.z));
        playerInfo.add("lookVector", lookVector);

        double horizontalLength = Math.sqrt(lookVec.x * lookVec.x + lookVec.z * lookVec.z);
        double dirX, dirZ;
        if (horizontalLength > 1.0e-6) {
            dirX = lookVec.x / horizontalLength;
            dirZ = lookVec.z / horizontalLength;
        } else {
            float yawRad = player.getYRot() * 0.017453292f;
            dirX = -Math.sin(yawRad);
            dirZ = Math.cos(yawRad);
        }

        Vec3 playerPos = new Vec3(player.getX(), player.getY(), player.getZ());
        Vec3 frontPos = new Vec3(playerPos.x + dirX * 3.0, playerPos.y, playerPos.z + dirZ * 3.0);
        JsonObject frontPosition = new JsonObject();
        frontPosition.addProperty("x", Integer.valueOf((int) Math.floor(frontPos.x)));
        frontPosition.addProperty("y", Integer.valueOf((int) Math.floor(frontPos.y)));
        frontPosition.addProperty("z", Integer.valueOf((int) Math.floor(frontPos.z)));
        playerInfo.add("frontPosition", frontPosition);

        playerInfo.addProperty("health", Float.valueOf(player.getHealth()));
        playerInfo.addProperty("maxHealth", Float.valueOf(player.getMaxHealth()));

        FoodData foodData = player.getFoodData();
        playerInfo.addProperty("foodLevel", Integer.valueOf(foodData.getFoodLevel()));
        playerInfo.addProperty("saturation", Float.valueOf(foodData.getSaturationLevel()));

        MultiPlayerGameMode gameMode = client.gameMode;
        if (gameMode != null) {
            playerInfo.addProperty("gameMode", gameMode.getPlayerMode().name().toLowerCase());
        }

        if (world != null) {
            playerInfo.addProperty("dimension", world.dimension().identifier().toString());
            playerInfo.addProperty("timeOfDay", Long.valueOf(world.getDefaultClockTime()));
            playerInfo.addProperty("isDay", Boolean.valueOf(world.isBrightOutside()));
            playerInfo.addProperty("isNight", Boolean.valueOf(world.isDarkOutside()));
            playerInfo.addProperty("isRaining", Boolean.valueOf(world.isRaining()));
        }

        playerInfo.addProperty("name", player.getName().getString());
        playerInfo.addProperty("experienceLevel", Integer.valueOf(player.experienceLevel));
        playerInfo.addProperty("experienceProgress", Float.valueOf(player.experienceProgress));
        playerInfo.addProperty("totalExperience", Integer.valueOf(player.totalExperience));

        JsonObject inventory = new JsonObject();
        Inventory inv = player.getInventory();
        inventory.addProperty("selectedSlot", Integer.valueOf(inv.getSelectedSlot()));

        ItemStack mainItem = player.getMainHandItem();
        inventory.addProperty("mainHandItem", mainItem.isEmpty() ? "empty" : mainItem.getItem().toString());

        ItemStack offItem = player.getOffhandItem();
        inventory.addProperty("offHandItem", offItem.isEmpty() ? "empty" : offItem.getItem().toString());

        playerInfo.add("inventory", inventory);

        com.google.gson.JsonArray nearbyEntities = new com.google.gson.JsonArray();
        if (world instanceof net.minecraft.client.multiplayer.ClientLevel clientLevel) {
            for (net.minecraft.world.entity.Entity entity : clientLevel.entitiesForRendering()) {
                if (entity == null || entity == player) continue;
                String typeStr = entity.getType().toShortString();
                if (typeStr.contains("item") || typeStr.contains("arrow") || typeStr.contains("experience_orb") || 
                    typeStr.contains("marker") || typeStr.contains("area_effect_cloud") || typeStr.contains("falling_block")) {
                    continue;
                }
                double ex = entity.getX();
                double ey = entity.getY();
                double ez = entity.getZ();
                double dx = ex - player.getX();
                double dy = ey - player.getY();
                double dz = ez - player.getZ();
                double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (dist <= 30.0) {
                    JsonObject entObj = new JsonObject();
                    entObj.addProperty("id", Integer.valueOf(entity.getId()));
                    entObj.addProperty("type", typeStr);
                    entObj.addProperty("name", entity.getName().getString());
                    entObj.addProperty("x", Double.valueOf(ex));
                    entObj.addProperty("y", Double.valueOf(ey));
                    entObj.addProperty("z", Double.valueOf(ez));
                    entObj.addProperty("dist", Double.valueOf(dist));
                    nearbyEntities.add(entObj);
                }
            }
        }
        playerInfo.add("nearbyEntities", nearbyEntities);

        return playerInfo;
    }

    private static String getCardinalDirection(float yaw) {
        yaw = yaw % 360.0f;
        if (yaw < 0) {
            yaw += 360.0f;
        }
        if (yaw >= 337.5 || yaw < 22.5) {
            return "South";
        }
        if (yaw >= 22.5 && yaw < 67.5) {
            return "Southwest";
        }
        if (yaw >= 67.5 && yaw < 112.5) {
            return "West";
        }
        if (yaw >= 112.5 && yaw < 157.5) {
            return "Northwest";
        }
        if (yaw >= 157.5 && yaw < 202.5) {
            return "North";
        }
        if (yaw >= 202.5 && yaw < 247.5) {
            return "Northeast";
        }
        if (yaw >= 247.5 && yaw < 292.5) {
            return "East";
        }
        return "Southeast";
    }
}
