#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;
out mat3 TBN;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoords = aTexCoords;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    FragPos = vec3(model * vec4(aPos, 1.0));

    
    vec3 T = normalize(mat3(model) * aTangent);
    vec3 B = normalize(mat3(model) * aBitangent);
    vec3 N = normalize(mat3(model) * aNormal);
    TBN = mat3(T, B, N);
}
)";

const char* fragmentShaderSource = R"(
    #version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;
in mat3 TBN;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_normal1;

uniform bool useNormalMap; 

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    vec3 normal;
    if (useNormalMap) {
        
        normal = texture(texture_normal1, TexCoords).rgb;
        normal = normalize(normal * 2.0 - 1.0); 
        normal = normalize(TBN * normal);      
    } else {
        
        normal = normalize(Normal);
    }

    
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);

    float ambientStrength = 0.5;
    float diff = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);

    vec3 ambient = ambientStrength * lightColor;
    vec3 diffuse = diff * lightColor;
    vec3 specular = 0.5 * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = texture(texture_diffuse1, TexCoords) * vec4(result, 1.0);
}
)";


const char* skyboxVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
    TexCoords = aPos;
}
)";

const char* skyboxFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{
    FragColor = texture(skybox, TexCoords);
}
)";

unsigned int loadTexture(const char* path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    } else {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

unsigned int loadCubemap(const std::unordered_map<std::string, std::string>& faces) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;

    
    std::unordered_map<std::string, GLenum> faceMapping = {
        {"right", GL_TEXTURE_CUBE_MAP_POSITIVE_X},
        {"left", GL_TEXTURE_CUBE_MAP_NEGATIVE_X},
        {"top", GL_TEXTURE_CUBE_MAP_POSITIVE_Y},
        {"bottom", GL_TEXTURE_CUBE_MAP_NEGATIVE_Y},
        {"front", GL_TEXTURE_CUBE_MAP_POSITIVE_Z},
        {"back", GL_TEXTURE_CUBE_MAP_NEGATIVE_Z}
    };

    for (const auto& [direction, path] : faces) {
        auto it = faceMapping.find(direction);
        if (it == faceMapping.end()) {
            std::cerr << "Unknown direction: " << direction << std::endl;
            continue;
        }

        unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            glTexImage2D(it->second, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        } else {
            std::cerr << "Cubemap texture failed to load at path: " << path << std::endl;
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}

unsigned int createShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource) {
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

std::vector<float> generateVertexTerrain(int* width, int* height, const char* heightMapPath) {
    std::vector<GLfloat> vertices;
    int nrComponents;
    unsigned char* heightMap = stbi_load(heightMapPath, width, height, &nrComponents, 0);

    for (int y = 0; y < *height; ++y) {
        for (int x = 0; x < *width; ++x) {
            int index = (y * *width + x) * nrComponents;
            GLfloat z = 0.0f;

            for (int i = 0; i < nrComponents; i++) {
                if (i < 3)
                    z += (GLfloat)heightMap[index + i] / 255.0f;

                if (i == 3)
                    z *= (GLfloat)heightMap[index + i];
            }

            z = 0.125 * z / (nrComponents <= 3 ? nrComponents : 3);

            glm::vec3 pos((GLfloat)x / *width - 0.5f, z, (GLfloat)y / *height - 0.5f);
            glm::vec2 texCoords((GLfloat)x / *width * 8, (GLfloat)y / *height * 8);

            
            glm::vec3 normal(0.0f, 1.0f, 0.0f);

            if (x > 0 && x < *width - 1 && y > 0 && y < *height - 1) {
                glm::vec3 left = glm::vec3(pos.x - 1.0f, heightMap[(y * *width + x - 1) * nrComponents], pos.z);
                glm::vec3 right = glm::vec3(pos.x + 1.0f, heightMap[(y * *width + x + 1) * nrComponents], pos.z);
                glm::vec3 up = glm::vec3(pos.x, heightMap[(y * *width + x) * nrComponents] + 1, pos.z + 1.0f);
                glm::vec3 down = glm::vec3(pos.x, heightMap[(y * *width + x) * nrComponents] - 1, pos.z - 1.0f);

                glm::vec3 tangent = right - left;
                glm::vec3 bitangent = up - down;
                normal = glm::normalize(glm::cross(tangent, bitangent));
            }

            vertices.push_back(pos.x);
            vertices.push_back(pos.y);
            vertices.push_back(pos.z);
            vertices.push_back(texCoords.x);
            vertices.push_back(texCoords.y);
            vertices.push_back(normal.x);
            vertices.push_back(normal.y);
            vertices.push_back(normal.z);
        }
    }

    stbi_image_free(heightMap);
    return vertices;
}

std::vector<unsigned int> generateIndexTerrain(int width, int height) {
    std::vector<unsigned int> indices;

    
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int index = y * width + x;
            indices.push_back(index);
            indices.push_back(index + width);
            indices.push_back(index + 1);

            indices.push_back(index + 1);
            indices.push_back(index + width);
            indices.push_back(index + width + 1);
        }
    }
    return indices;
}

class Model {
public:
    Model(const char* path, unsigned int textureID, unsigned int normalMapID = 0) {
        this->textureID = textureID;
        this->normalMapID = normalMapID;
        this->hasNormalMap = (normalMapID != 0);

        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
            return;
        }
        processNode(scene->mRootNode, scene);
    }

    void Draw(GLuint shaderProgram) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, "texture_diffuse1"), 0);

        if (hasNormalMap) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, normalMapID);
            glUniform1i(glGetUniformLocation(shaderProgram, "texture_normal1"), 1);
        }

        glUniform1i(glGetUniformLocation(shaderProgram, "useNormalMap"), hasNormalMap);

        for (auto& mesh : meshes) {
            mesh.Draw(shaderProgram);
        }
    }

private:
    bool hasNormalMap;
    unsigned int textureID;
    unsigned int normalMapID;
    struct Vertex {
        glm::vec3 position;
        glm::vec2 texCoords;
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec3 bitangent;
    };

    struct Mesh {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        unsigned int VAO;

        Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices) {
            this->vertices = vertices;
            this->indices = indices;
            setupMesh();
        }

        void Draw(GLuint shaderProgram) {
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        void setupMesh() {
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);

            glBindVertexArray(VAO);

            
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

            
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
            glEnableVertexAttribArray(0);

            
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
            glEnableVertexAttribArray(1);

            
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
            glEnableVertexAttribArray(2);

            
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));
            glEnableVertexAttribArray(3);

            
            glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));
            glEnableVertexAttribArray(4);

            glBindVertexArray(0);
        }

        unsigned int VBO, EBO;
    };

    std::vector<Mesh> meshes;

    void processNode(aiNode* node, const aiScene* scene) {
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }

        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene);
        }
    }

    Mesh processMesh(aiMesh* mesh, const aiScene* scene) {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Vertex vertex;

            
            vertex.position.x = mesh->mVertices[i].x;
            vertex.position.y = mesh->mVertices[i].y;
            vertex.position.z = mesh->mVertices[i].z;

            
            if (mesh->HasNormals()) {
                vertex.normal.x = mesh->mNormals[i].x;
                vertex.normal.y = mesh->mNormals[i].y;
                vertex.normal.z = mesh->mNormals[i].z;
            } else {
                vertex.normal = glm::vec3(0.0f, 0.0f, 0.0f); 
            }

            
            if (mesh->mTextureCoords[0]) { 
                vertex.texCoords.x = mesh->mTextureCoords[0][i].x;
                vertex.texCoords.y = mesh->mTextureCoords[0][i].y;
            } else {
                vertex.texCoords = glm::vec2(0.0f, 0.0f); 
            }

            
            vertex.tangent = glm::vec3(0.0f);
            vertex.bitangent = glm::vec3(0.0f);

            vertices.push_back(vertex);
        }

        
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }

            
            if (face.mNumIndices == 3) { 
                Vertex& v0 = vertices[face.mIndices[0]];
                Vertex& v1 = vertices[face.mIndices[1]];
                Vertex& v2 = vertices[face.mIndices[2]];

                glm::vec3 edge1 = v1.position - v0.position;
                glm::vec3 edge2 = v2.position - v0.position;

                glm::vec2 deltaUV1 = v1.texCoords - v0.texCoords;
                glm::vec2 deltaUV2 = v2.texCoords - v0.texCoords;

                float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

                glm::vec3 tangent;
                tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
                tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
                tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
                tangent = glm::normalize(tangent);

                glm::vec3 bitangent;
                bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
                bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
                bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
                bitangent = glm::normalize(bitangent);

                v0.tangent += tangent;
                v1.tangent += tangent;
                v2.tangent += tangent;

                v0.bitangent += bitangent;
                v1.bitangent += bitangent;
                v2.bitangent += bitangent;
            }
        }

        
        for (auto& vertex : vertices) {
            vertex.tangent = glm::normalize(vertex.tangent);
            vertex.bitangent = glm::normalize(vertex.bitangent);
        }

        return Mesh(vertices, indices);
    }
};


int main() {

    
    std::string landscapeTexturePath = std::getenv("LANDSCAPE_TEXTURE_PATH");
    std::string heightMapPath = std::getenv("HEIGHT_MAP_PATH");

    std::unordered_map<std::string, std::string> faces = {
        {"right", std::getenv("SKYBOX_RIGHT_PATH")},
        {"left", std::getenv("SKYBOX_LEFT_PATH")},
        {"top", std::getenv("SKYBOX_TOP_PATH")},
        {"bottom", std::getenv("SKYBOX_BOTTOM_PATH")},
        {"front", std::getenv("SKYBOX_FRONT_PATH")},
        {"back", std::getenv("SKYBOX_BACK_PATH")}
    };

    std::string model1Path = std::getenv("MODEL1_PATH");
    std::string model1TexturePath = std::getenv("MODEL1_TEXTURE_PATH");
    std::string model1NormalMapPath = std::getenv("MODEL1_NORMAL_MAP_PATH");

    std::string model2Path = std::getenv("MODEL2_PATH");
    std::string model2TexturePath = std::getenv("MODEL2_TEXTURE_PATH");

    std::string model3Path = std::getenv("MODEL3_PATH");
    std::string model3TexturePath = std::getenv("MODEL3_TEXTURE_PATH");
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Landscape with Skybox and Lighting", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    
    unsigned int landscapeShaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    glUseProgram(landscapeShaderProgram);

    
    unsigned int skyboxShaderProgram = createShaderProgram(skyboxVertexShaderSource, skyboxFragmentShaderSource);
    glUseProgram(skyboxShaderProgram);

    
    unsigned int landscapeTexture = loadTexture(landscapeTexturePath.c_str());

    unsigned int skyboxTexture = loadCubemap(faces);

    
    int terrainWidth;
    int terrainHeight;
    std::vector<float> vertices = generateVertexTerrain(&terrainWidth, &terrainHeight, heightMapPath.c_str());
    std::vector<unsigned int> indices = generateIndexTerrain(terrainWidth, terrainHeight);

    
    unsigned int landscapeVAO, landscapeVBO, landscapeEBO;
    glGenVertexArrays(1, &landscapeVAO);
    glGenBuffers(1, &landscapeVBO);
    glGenBuffers(1, &landscapeEBO);

    glBindVertexArray(landscapeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, landscapeVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, landscapeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);

    glBindVertexArray(skyboxVAO);

    float skyboxVertices[] = {
        -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

    -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f
    };

    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    Model ourModel1(model1Path.c_str(), loadTexture(model1TexturePath.c_str()), loadTexture(model1NormalMapPath.c_str()));

    Model ourModel2(model2Path.c_str(), loadTexture(model2TexturePath.c_str()));

    Model ourModel3(model3Path.c_str(), loadTexture(model3TexturePath.c_str()));

    
    glm::vec3 cameraPos = glm::vec3(0.0f, 5.0f, 10.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    float yaw = -90.0f;
    float pitch = 0.0f;
    float lastX = 400, lastY = 300;
    float cameraSpeed = 0.05f;

    
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    
    glm::vec3 lightPos(1.0f, 1.0f, 1.0f);
    glm::vec3 viewPos = cameraPos;
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor(1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    
    while (!glfwWindowShouldClose(window)) {
        
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

        
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; 
        lastX = xpos;
        lastY = ypos;

        float sensitivity = 0.2f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);

        
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        
        glUseProgram(landscapeShaderProgram);
        glm::mat4 landscapeModel = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(landscapeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(landscapeModel));
        glUniformMatrix4fv(glGetUniformLocation(landscapeShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(landscapeShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        
        glUniform3f(glGetUniformLocation(landscapeShaderProgram, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
        glUniform3f(glGetUniformLocation(landscapeShaderProgram, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
        glUniform3f(glGetUniformLocation(landscapeShaderProgram, "lightColor"), lightColor.x, lightColor.y, lightColor.z);
        glUniform3f(glGetUniformLocation(landscapeShaderProgram, "objectColor"), objectColor.x, objectColor.y, objectColor.z);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, landscapeTexture);
        glBindVertexArray(landscapeVAO);
        glDrawElements(GL_TRIANGLES, terrainWidth * terrainHeight * 6, GL_UNSIGNED_INT, NULL);

        
        glm::mat4 model1 = glm::mat4(1.0f);
        model1 = glm::translate(model1, glm::vec3(0.0f, 0.125f, 0.0f)); 
        model1 = glm::scale(model1, glm::vec3(0.005f)); 
        model1 = glm::rotate(model1, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); 
        glUniformMatrix4fv(glGetUniformLocation(landscapeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model1));
        ourModel1.Draw(landscapeShaderProgram);

        
        glm::mat4 model2 = glm::mat4(1.0f);
        model2 = glm::translate(model2, glm::vec3(-0.125f, 0.125f, 0.0f)); 
        model2 = glm::scale(model2, glm::vec3(0.005f)); 
        model2 = glm::rotate(model2, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); 
        model2 = glm::rotate(model2, glm::radians(-90.0f), glm::vec3(0.0f, 0.0f, 1.0f)); 
        glUniformMatrix4fv(glGetUniformLocation(landscapeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model2));
        ourModel2.Draw(landscapeShaderProgram);

        
        glm::mat4 model3 = glm::mat4(1.0f);
        model3 = glm::translate(model3, glm::vec3(0.125f, 0.125f, 0.0f)); 
        model3 = glm::scale(model3, glm::vec3(0.0051f)); 
        model3 = glm::rotate(model3, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); 
        glUniformMatrix4fv(glGetUniformLocation(landscapeShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model3));
        ourModel3.Draw(landscapeShaderProgram);

        
        glDepthFunc(GL_LEQUAL); 
        glUseProgram(skyboxShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(glm::mat4(glm::mat3(view))));
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glDepthFunc(GL_LESS); 

        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    
    glDeleteVertexArrays(1, &landscapeVAO);
    glDeleteBuffers(1, &landscapeVBO);
    glDeleteVertexArrays(1, &skyboxVAO);
    glDeleteBuffers(1, &skyboxVBO);
    glDeleteProgram(landscapeShaderProgram);
    glDeleteProgram(skyboxShaderProgram);

    glfwTerminate();
    return 0;
}
